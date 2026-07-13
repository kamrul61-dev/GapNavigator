import pytest
from app.agents.state import AgentState
from app.agents.workflow import get_workflow_graph
from app.agents.resume_agent import resume_agent_node
from app.agents.gap_agent import gap_agent_node
from app.agents.roadmap_agent import roadmap_agent_node

def test_workflow_compiles():
    """Verify that the LangGraph workflow compiles and contains all expected nodes."""
    graph = get_workflow_graph()
    assert graph is not None
    
    # In LangGraph compiled apps, the nodes dictionary lists all configured nodes
    node_names = list(graph.nodes.keys())
    assert "resume_agent" in node_names
    assert "gap_agent" in node_names
    assert "roadmap_agent" in node_names

def test_resume_agent_empty_input_handling():
    """Verify the resume agent node logs an error if the input resume text is empty."""
    state: AgentState = {
        "resume_text": "",
        "job_description": "Python Software Engineer",
        "resume_data": None,
        "gap_analysis": None,
        "roadmap": None,
        "retrieved_resources": [],
        "errors": []
    }
    
    output_state = resume_agent_node(state)
    assert output_state["resume_data"] is None
    assert len(output_state["errors"]) == 1
    assert "empty or missing" in output_state["errors"][0]

def test_gap_agent_missing_dependencies():
    """Verify that the gap agent registers an error if previous nodes did not set resume_data."""
    state: AgentState = {
        "resume_text": "Experienced coder",
        "job_description": "Python Software Engineer",
        "resume_data": None,  # Missing!
        "gap_analysis": None,
        "roadmap": None,
        "retrieved_resources": [],
        "errors": []
    }
    
    output_state = gap_agent_node(state)
    assert output_state["gap_analysis"] is None
    assert len(output_state["errors"]) == 1
    assert "resume data is missing" in output_state["errors"][0]

def test_roadmap_agent_missing_dependencies():
    """Verify that the roadmap agent registers an error if previous nodes did not set gap_analysis."""
    state: AgentState = {
        "resume_text": "Experienced coder",
        "job_description": "Python Software Engineer",
        "resume_data": None,
        "gap_analysis": None,  # Missing!
        "roadmap": None,
        "retrieved_resources": [],
        "errors": []
    }
    
    output_state = roadmap_agent_node(state)
    assert output_state["roadmap"] is None
    assert len(output_state["errors"]) == 1
    assert "gap analysis results are missing" in output_state["errors"][0]
