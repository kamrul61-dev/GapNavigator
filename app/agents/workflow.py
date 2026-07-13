import logging
from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.resume_agent import resume_agent_node
from app.agents.gap_agent import gap_agent_node
from app.agents.roadmap_agent import roadmap_agent_node

logger = logging.getLogger("gapnavigator.agents.workflow")

def get_workflow_graph():
    """
    Constructs and compiles the LangGraph StateGraph workflow.
    
    Flow: Start -> resume_agent -> gap_agent -> roadmap_agent -> End
    """
    # 1. Define a state graph with the schema
    workflow = StateGraph(AgentState)
    
    # 2. Add the agent node functions
    workflow.add_node("resume_agent", resume_agent_node)
    workflow.add_node("gap_agent", gap_agent_node)
    workflow.add_node("roadmap_agent", roadmap_agent_node)
    
    # 3. Define the edges
    workflow.set_entry_point("resume_agent")
    
    # Linear connections
    workflow.add_edge("resume_agent", "gap_agent")
    workflow.add_edge("gap_agent", "roadmap_agent")
    workflow.add_edge("roadmap_agent", END)
    
    # 4. Compile the graph
    app = workflow.compile()
    return app

def run_career_planner_workflow(resume_text: str, job_description: str) -> dict:
    """
    Convenience orchestrator function to run the entire planning graph.
    
    Args:
        resume_text: Raw plain text extracted from candidate's resume
        job_description: Target job description pasted by user
        
    Returns:
        dict: The final workflow execution state dictionary
    """
    logger.info("Initializing LangGraph Career Planner execution...")
    
    initial_state = {
        "resume_text": resume_text,
        "job_description": job_description,
        "resume_data": None,
        "gap_analysis": None,
        "roadmap": None,
        "retrieved_resources": [],
        "errors": []
    }
    
    graph = get_workflow_graph()
    
    # Run graph execution to completion
    result = graph.invoke(initial_state)
    return result
