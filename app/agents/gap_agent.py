import os
import json
import logging
from app.agents.state import AgentState
from app.agents.resume_agent import get_llm
from app.models.schemas import GapAnalysis

logger = logging.getLogger("gapnavigator.agents.gap_agent")

def gap_agent_node(state: AgentState) -> AgentState:
    """
    Compares the candidate's structured resume skills/qualifications 
    against the target job description to compute a gap analysis.
    """
    logger.info("Executing Gap Analysis Agent Node...")
    errors = state.get("errors", [])[:]
    
    # Check dependencies
    if not state.get("resume_data"):
        err_msg = "Cannot run Gap Analysis because resume data is missing."
        logger.error(err_msg)
        errors.append(f"Gap Agent Error: {err_msg}")
        state["errors"] = errors
        return state
        
    if not state.get("job_description"):
        err_msg = "Job description is empty or missing."
        logger.error(err_msg)
        errors.append(f"Gap Agent Error: {err_msg}")
        state["errors"] = errors
        return state

    try:
        # Load prompt template
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts",
            "gap_analysis.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
            
        llm = get_llm()
        structured_llm = llm.with_structured_output(GapAnalysis)
        
        # Serialize candidate profile to JSON for readability by LLM
        candidate_profile_json = json.dumps(state["resume_data"].dict(), indent=2)
        
        messages = [
            ("system", system_prompt),
            ("human", (
                f"Candidate Structured Resume Profile:\n{candidate_profile_json}\n\n"
                f"Target Job Description:\n{state['job_description']}"
            ))
        ]
        
        response = structured_llm.invoke(messages)
        state["gap_analysis"] = response
        
    except Exception as e:
        err_msg = f"Failed to perform gap analysis with Gemini: {e}"
        logger.error(err_msg)
        errors.append(f"Gap Agent Error: {err_msg}")
        state["errors"] = errors
        state["gap_analysis"] = None
        
    return state
