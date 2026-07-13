from typing import TypedDict, Optional, List
from app.models.schemas import ResumeData, GapAnalysis, Roadmap

class AgentState(TypedDict):
    resume_text: str
    job_description: str
    resume_data: Optional[ResumeData]
    gap_analysis: Optional[GapAnalysis]
    roadmap: Optional[Roadmap]
    retrieved_resources: List[str]
    errors: List[str]
