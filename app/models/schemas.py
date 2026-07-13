from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Skill(BaseModel):
    name: str = Field(description="Name of the skill, e.g. Python, SQL, Project Management")
    category: Literal["technical", "soft"] = Field(description="Categorize skill as either 'technical' or 'soft'")

class EducationItem(BaseModel):
    degree: str = Field(description="Name of the degree or certification, e.g. B.S. in Computer Science")
    institution: str = Field(description="Name of the school, university, or issuing organization")
    field_of_study: Optional[str] = Field(None, description="Field of study or major, if applicable")
    graduation_year: Optional[str] = Field(None, description="Graduation or completion year, e.g. 2024")

class ExperienceItem(BaseModel):
    job_title: str = Field(description="Role or job title, e.g. Software Engineer Intern")
    company: str = Field(description="Company or organization name")
    duration: Optional[str] = Field(None, description="Period of employment, e.g. Jun 2022 - Aug 2023 or 2 years")
    responsibilities: List[str] = Field(default_factory=list, description="List of key duties, achievements, or projects")

class ResumeData(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address of the candidate")
    phone: str = Field(description="Phone number of the candidate")
    skills: List[Skill] = Field(default_factory=list, description="List of technical and soft skills extracted from resume")
    education: List[EducationItem] = Field(default_factory=list, description="Education details extracted from resume")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Work experience details extracted from resume")

class GapAnalysis(BaseModel):
    existing_skills: List[Skill] = Field(
        default_factory=list, 
        description="Skills candidate possesses that are relevant to the job"
    )
    missing_skills: List[Skill] = Field(
        default_factory=list, 
        description="Key skills required by the job but missing or weak in the resume"
    )
    suggested_skills: List[Skill] = Field(
        default_factory=list, 
        description="Additional skills that would be beneficial or nice-to-have for this job"
    )
    readiness_score: int = Field(
        description="Job readiness percentage from 0 to 100 based on skill match and qualifications"
    )
    strengths: List[str] = Field(
        default_factory=list, 
        description="Key strengths of the candidate relative to the job requirements"
    )
    weaknesses: List[str] = Field(
        default_factory=list, 
        description="Key skill gaps or areas of improvement for the candidate"
    )
    explanation: str = Field(
        description="A brief, professional summary explaining the gap analysis and readiness score"
    )

class LearningResource(BaseModel):
    title: str = Field(description="Name or title of the learning resource, course, or tutorial")
    url: str = Field(description="Web link/URL to the resource")
    resource_type: str = Field(description="Type of resource, e.g. Documentation, Course, YouTube Playlist, Tutorial")
    description: str = Field(description="Brief summary of what this resource covers and how it helps bridge the gap")

class RoadmapItem(BaseModel):
    week: int = Field(description="The week number, e.g. 1, 2, 3, 4")
    goals: List[str] = Field(description="Main learning goals for this week")
    topics: List[str] = Field(description="Specific topics to study")
    practice_tasks: List[str] = Field(description="Practical tasks or project milestones to build hands-on skills")

class Roadmap(BaseModel):
    weekly_plan: List[RoadmapItem] = Field(description="A weekly study plan spanning exactly 4 weeks (30 days)")
    recommended_resources: List[LearningResource] = Field(
        default_factory=list, 
        description="Real learning resources recommended to help with the weekly study goals"
    )
