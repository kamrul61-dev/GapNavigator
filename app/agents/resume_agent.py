import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agents.state import AgentState
from app.models.schemas import ResumeData

logger = logging.getLogger("gapnavigator.agents.resume_agent")

def get_llm():
    """Helper to initialize the Google Generative AI LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0
    )

def resume_agent_node(state: AgentState) -> AgentState:
    """
    Analyzes resume text and extracts structured information 
    including contact details, skills, education, and experience.
    """
    logger.info("Executing Resume Agent Node...")
    errors = state.get("errors", [])[:]
    
    # Check for input
    if not state.get("resume_text"):
        err_msg = "Resume text is empty or missing."
        logger.error(err_msg)
        errors.append(f"Resume Agent Error: {err_msg}")
        state["errors"] = errors
        return state

    try:
        # Load prompt template
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts",
            "resume_extract.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
            
        llm = get_llm()
        structured_llm = llm.with_structured_output(ResumeData)
        
        messages = [
            ("system", system_prompt),
            ("human", f"Please parse this resume text:\n\n{state['resume_text']}")
        ]
        
        response = structured_llm.invoke(messages)
        state["resume_data"] = response
        
    except Exception as e:
        err_msg = f"Failed to parse resume text with Gemini: {e}"
        logger.error(err_msg)
        errors.append(f"Resume Agent Error: {err_msg}")
        state["errors"] = errors
        state["resume_data"] = None
        
    return state
