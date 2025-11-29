import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Any, Dict

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Agent Imports ---
from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
)

# ----------------------------------------------------------------------
# 1. Configuration and Global Setup
# ----------------------------------------------------------------------
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_base_url = os.getenv("GEMINI_BASE_URL")

if not gemini_api_key:
    print("Warning: GEMINI_API_KEY not found in .env. Using a placeholder.")
    gemini_api_key = "placeholder-key"

# Initialize the external client and model
external_client = AsyncOpenAI(api_key=gemini_api_key, base_url=gemini_base_url)
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=external_client)
config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

# ----------------------------------------------------------------------
# 2. Agent Output and Guardrail Definition
# ----------------------------------------------------------------------
class FitAgentOutput(BaseModel):
    """Structured output for the Guardrail Agent decision."""
    is_fit_agent: bool
    reasoning: str

# Guardrail agent: determines if input is fitness-related
guardrail_agent = Agent(
    name="Guardrail check",
    instructions=(
         "Decide whether the user's message is about the content of the book. "
    "Return JSON exactly in this format: "
    '{"is_book_agent": true/false, "reasoning": "short explanation"}. '
    "Return is_book_agent = true ONLY when the user is clearly asking for questions, explanations, "
    "or clarifications related to the book chapters, topics, or examples. "
    "If the input is violent, illegal, sexual, political, unrelated to the book, or about external topics, return false."

    ),
    model=model,
    output_type=FitAgentOutput,
)

@input_guardrail
async def fitness_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """
    Run the guardrail agent and return a GuardrailFunctionOutput.
    Conservative default: block unless guardrail explicitly returns is_fit_agent = True.
    """
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    print("DEBUG: Guardrail raw result:", getattr(result, "final_output", None))

    trip = True  # default: block unless we see boolean True
    output_info = {"is_fit_agent": False, "reasoning": "Unknown / malformed output."}

    if getattr(result, "final_output", None) is not None:
        try:
            is_fit = result.final_output.is_fit_agent
            reasoning = result.final_output.reasoning
            if isinstance(is_fit, bool):
                output_info = {"is_fit_agent": is_fit, "reasoning": reasoning}
                trip = not is_fit
            else:
                output_info["reasoning"] = "Guardrail output 'is_fit_agent' is not boolean."
        except Exception as e:
            output_info["reasoning"] = f"Exception parsing guardrail output: {e}"
    else:
        output_info["reasoning"] = "Guardrail returned no final_output."

    print("DEBUG: tripwire =", trip, "| output_info =", output_info)
    return GuardrailFunctionOutput(output_info=output_info, tripwire_triggered=trip)

# Main assistant agent using the input guardrail
assistant_agent = Agent(
    name="Assistant",
    instructions="You are a helpful book assistant. Only respond to questions related to the book content, chapters, examples, or selected text. Provide accurate, grounded answers with source references whenever possible. If the question is unrelated to the book, politely decline to answer."
    
    
,
    model=model,
    input_guardrails=[fitness_guardrail],
)

# ----------------------------------------------------------------------
# 3. FastAPI Setup
# ----------------------------------------------------------------------
app = FastAPI(
    title="Gemini Agent API",
    description="Backend service for a Guardrail-protected Gemini Agent",
    version="1.0.0",
)

# CORS - allow all origins for development. Restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatRequest(BaseModel):
    message: str

# ----------------------------------------------------------------------
# 4. Utility to normalize response_text
# ----------------------------------------------------------------------
def normalize_final_output(final_output: Any) -> str:
    """
    Convert agent final_output (which may be string, pydantic model, or other) into a safe string.
    """
    if final_output is None:
        return ""
    # If it's already a simple type, stringify
    if isinstance(final_output, (str, int, float, bool)):
        return str(final_output)
    # If it's a pydantic model (BaseModel), return its text-like attributes if present, else JSON
    try:
        if isinstance(final_output, BaseModel):
            # Prefer fields named 'text', 'response', 'output', else full json
            data = final_output.dict()
            for candidate in ("text", "response", "output", "final_output"):
                if candidate in data and isinstance(data[candidate], (str, int, float, bool)):
                    return str(data[candidate])
            # fallback to JSON representation
            return final_output.json()
    except Exception:
        pass
    # last resort: stringify the object
    try:
        return str(final_output)
    except Exception:
        return "Unable to decode agent output."

# ----------------------------------------------------------------------
# 5. Endpoint
# ----------------------------------------------------------------------
@app.post("/chat", summary="Process a user message through the Guardrail Agent")
async def chat_endpoint(request: ChatRequest = Body(...)) -> Dict[str, Any]:
    user_input = request.message
    print(f"Received user message: {user_input}")

    try:
        # Run the assistant agent - if guardrail trips, InputGuardrailTripwireTriggered will be raised
        res = await Runner.run(assistant_agent, user_input, run_config=config)

        # Normalize the agent's final output to a string
        final_output_raw = getattr(res, "final_output", None)
        response_text = normalize_final_output(final_output_raw)

        return {
            "status": "success",
            "message": "Agent response generated.",
            "response_text": response_text,
        }

    except InputGuardrailTripwireTriggered as e:
        # Guardrail blocked the request - extract reasoning safely
        output_info = getattr(e, "output_info", None)
        # output_info may be dict-like or pydantic or other
        reason_text = ""
        try:
            if isinstance(output_info, dict):
                reason_text = output_info.get("reasoning") or output_info.get("reason") or ""
            elif hasattr(output_info, "reasoning"):
                reason_text = getattr(output_info, "reasoning", "")
            else:
                reason_text = str(output_info)
        except Exception:
            reason_text = ""

        polite_refusal = "⚠️ Sorry, I cannot help with that. Only fitness-related questions are allowed."
        if reason_text:
            polite_refusal += f" Reason: {reason_text}"

        # Always return 200 JSON (frontend expects JSON); do not raise an exception here
        return {
            "status": "blocked",
            "message": "Request blocked by the fitness guardrail.",
            "response_text": polite_refusal,
        }

    except Exception as e:
        # Unexpected error - log and return a controlled error response
        print(f"An unexpected error occurred: {e}")
        # Return a JSON error response with 500 status
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# ----------------------------------------------------------------------
# 6. Server execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting FastAPI server on host 0.0.0.0 and port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)