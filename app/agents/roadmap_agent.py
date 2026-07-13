import os
import json
import logging
from app.agents.state import AgentState
from app.agents.resume_agent import get_llm
from app.rag.retriever import retrieve_resources
from app.models.schemas import Roadmap

logger = logging.getLogger("gapnavigator.agents.roadmap_agent")

def roadmap_agent_node(state: AgentState) -> AgentState:
    """
    Retrieves learning materials for the missing skills,
    then generates a structured 30-day weekly learning roadmap and course list.
    """
    logger.info("Executing Roadmap Agent Node...")
    errors = state.get("errors", [])[:]
    
    # Check dependencies
    if not state.get("gap_analysis"):
        err_msg = "Cannot run Roadmap Agent because gap analysis results are missing."
        logger.error(err_msg)
        errors.append(f"Roadmap Agent Error: {err_msg}")
        state["errors"] = errors
        return state
        
    try:
        gap_data = state["gap_analysis"]
        missing_skills_names = [s.name for s in gap_data.missing_skills]
        existing_skills_names = [s.name for s in gap_data.existing_skills]
        
        # 1. Retrieve relevant courses from knowledge base using missing skills
        if missing_skills_names:
            query = ", ".join(missing_skills_names)
            logger.info(f"Querying FAISS vector store with missing skills: {query}")
        else:
            # Fallback if there are no missing skills
            query = state.get("job_description", "")[:200]
            logger.info("No missing skills found. Querying FAISS vector store with job description.")
            
        retrieved_docs = retrieve_resources(query, k=5)
        state["retrieved_resources"] = retrieved_docs
        
        # Format the retrieved documents for the LLM prompt
        retrieved_context = "\n---\n".join(retrieved_docs) if retrieved_docs else "No specific learning materials found in the knowledge base."
        
        # 2. Load prompt template
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts",
            "roadmap_gen.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
            
        llm = get_llm()
        structured_llm = llm.with_structured_output(Roadmap)
        
        # Construct message content
        human_content = (
            f"Candidate Existing Skills: {', '.join(existing_skills_names)}\n"
            f"Candidate Missing Skills to Learn: {', '.join(missing_skills_names)}\n\n"
            f"Retrieved Learning Resources from Knowledge Base:\n{retrieved_context}"
        )
        
        messages = [
            ("system", system_prompt),
            ("human", human_content)
        ]
        
        response = structured_llm.invoke(messages)
        state["roadmap"] = response
        
    except Exception as e:
        err_msg = f"Failed to generate learning roadmap with Gemini: {e}"
        logger.error(err_msg)
        errors.append(f"Roadmap Agent Error: {err_msg}")
        state["errors"] = errors
        state["roadmap"] = None
        
    return state
