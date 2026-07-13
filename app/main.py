import os
import sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Ensure the root folder is on the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.logger import setup_logger
from app.parsers.pdf_parser import extract_text_from_pdf, PDFParsingError
from app.agents.workflow import run_career_planner_workflow

# 1. Load env and setup logger
load_dotenv()
logger = setup_logger()

# 2. Page Configuration
st.set_page_config(
    page_title="GapNavigator Lite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Custom Styling (CSS Injection)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background & Title Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(42, 82, 152, 0.15);
        text-align: center;
    }
    
    .header-banner h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    
    .header-banner p {
        font-size: 1.15rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Card Layouts */
    .premium-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(220, 224, 230, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 1.25rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .premium-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.04);
    }
    
    /* Readiness Score Gauge Styles */
    .score-container {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 50%;
        width: 140px;
        height: 140px;
        margin: auto;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
        border: 4px solid #f1f3f6;
    }
    
    .score-value {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.1;
    }
    
    .score-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #7a828e;
        margin-top: 4px;
    }
    
    /* Custom Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    .badge-existing {
        background-color: #e2f9f0;
        color: #10b981;
        border: 1px solid #a7f3d0;
    }
    
    .badge-missing {
        background-color: #fff1f2;
        color: #f43f5e;
        border: 1px solid #fecdd3;
    }
    
    .badge-suggested {
        background-color: #eef2ff;
        color: #6366f1;
        border: 1px solid #e0e7ff;
    }
    
    /* Roadmap Weeks */
    .roadmap-week-header {
        color: #1e3c72;
        font-weight: 600;
        border-bottom: 2px solid #eef2ff;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    /* Course Resource Card */
    .resource-card {
        background-color: #fafbfc;
        border-left: 4px solid #6366f1;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.75rem;
    }
    .resource-title a {
        color: #2a5298;
        font-weight: 600;
        text-decoration: none;
    }
    .resource-title a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# 4. Header Banner
st.markdown("""
    <div class="header-banner">
        <h1>🎯 GapNavigator Lite</h1>
        <p>Analyze resume skill gaps, compute job readiness, and generate a 30-day personalized learning roadmap.</p>
    </div>
""", unsafe_allow_html=True)

# 5. Sidebar - API configuration & Info
st.sidebar.header("🔑 Credentials Configuration")

# Check if environment key exists, otherwise enable sidebar fields
env_google_key = os.getenv("GOOGLE_API_KEY", "")
api_key_provided = True

if not env_google_key:
    google_key_input = st.sidebar.text_input(
        "Google API Key:", 
        type="password",
        help="Required to use Google Gemini Flash models."
    )
    if google_key_input:
        os.environ["GOOGLE_API_KEY"] = google_key_input
        api_key_provided = True
    else:
        st.sidebar.error("⚠️ GOOGLE_API_KEY is required to run the AI workflow.")
        api_key_provided = False
else:
    st.sidebar.success("✅ Google API Key loaded from `.env`")
    api_key_provided = True

# Tracing info
langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
if langsmith_tracing:
    st.sidebar.success("✅ LangSmith Tracing Active")
else:
    st.sidebar.info("💡 LangSmith Tracing is inactive (check `.env` to enable)")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### How to Use:
1. Upload your Resume in **PDF** format.
2. Paste the **Job Description** of your target role.
3. Click **Analyze Job Fit**.
4. Review your Skill Gaps, Readiness Score, and your 30-Day Learning Plan!
""")

# 6. Main Inputs Page Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
    
with col2:
    st.subheader("💼 Target Job Description")
    job_desc_input = st.text_area(
        "Paste the target job description here...", 
        height=180,
        placeholder="e.g. We are looking for a Python Software Engineer with experience in Streamlit, SQL, Git, and REST APIs..."
    )

st.markdown("<br>", unsafe_allow_html=True)

# 7. Execution and Trigger Button
analyze_clicked = st.button("🚀 Analyze Job Fit & Generate Roadmap", use_container_width=True)

if analyze_clicked:
    if not api_key_provided:
        st.error("Please provide your Google API Key in the sidebar or in your `.env` file before executing.")
    elif not uploaded_file:
        st.warning("Please upload a resume PDF file.")
    elif not job_desc_input.strip():
        st.warning("Please paste a target job description.")
    else:
        # Run workflow with spinner
        with st.spinner("Parsing resume and analyzing skill gaps..."):
            logger.info("Starting manual user-triggered analysis...")
            
            # Step 1: Parse PDF
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
                st.toast("Resume parsed successfully!", icon="✅")
            except PDFParsingError as pe:
                st.error(f"PDF Parsing Failed: {pe}")
                logger.error(f"User PDF upload parse failed: {pe}")
                st.stop()
            except Exception as e:
                st.error(f"An unexpected parsing error occurred: {e}")
                st.stop()
                
            # Step 2: Invoke LangGraph Agent workflow
            with st.spinner("AI Agents executing gap comparison & roadmap recommendations..."):
                try:
                    result = run_career_planner_workflow(resume_text, job_desc_input)
                    
                    if result.get("errors"):
                        for error in result["errors"]:
                            st.error(error)
                        st.stop()
                        
                    # Extract variables
                    resume_data = result.get("resume_data")
                    gap_analysis = result.get("gap_analysis")
                    roadmap = result.get("roadmap")
                    
                    if not resume_data or not gap_analysis or not roadmap:
                        st.error("Agents failed to return structured outputs. Please verify your Gemini API key & settings.")
                        st.stop()
                        
                    # 8. Display Results Page
                    st.success("Analysis Complete!")
                    st.markdown("---")
                    
                    # Layout output tabs
                    tab_summary, tab_gap, tab_roadmap = st.tabs([
                        "📄 Resume Profile", 
                        "⚖️ Gap Analysis & Job Match", 
                        "📅 30-Day Learning Plan"
                    ])
                    
                    # Tab 1: Resume Profile Summary
                    with tab_summary:
                        st.markdown(f"### Candidate Profile: **{resume_data.name}**")
                        c_col1, c_col2 = st.columns(2)
                        with c_col1:
                            st.write(f"📧 **Email:** {resume_data.email}")
                        with c_col2:
                            st.write(f"📞 **Phone:** {resume_data.phone}")
                            
                        st.markdown("#### 🎓 Education")
                        if resume_data.education:
                            for edu in resume_data.education:
                                grad_yr = f"({edu.graduation_year})" if edu.graduation_year else ""
                                major = f" in {edu.field_of_study}" if edu.field_of_study else ""
                                st.markdown(f"- **{edu.degree}**{major} – *{edu.institution}* {grad_yr}")
                        else:
                            st.info("No education entries extracted.")
                            
                        st.markdown("#### 💼 Professional Experience")
                        if resume_data.experience:
                            for exp in resume_data.experience:
                                duration_str = f"({exp.duration})" if exp.duration else ""
                                st.markdown(f"**{exp.job_title}** at *{exp.company}* {duration_str}")
                                for resp in exp.responsibilities:
                                    st.markdown(f"  - {resp}")
                        else:
                            st.info("No professional experience entries extracted.")
                            
                    # Tab 2: Gap Analysis & Job Match
                    with tab_gap:
                        score = gap_analysis.readiness_score
                        
                        # Color coding readiness score
                        if score >= 80:
                            score_color = "#10b981"  # Emerald Green
                            rating = "Strong Fit"
                        elif score >= 60:
                            score_color = "#f59e0b"  # Amber Yellow
                            rating = "Moderate Fit"
                        else:
                            score_color = "#ef4444"  # Red
                            rating = "Development Needed"
                            
                        m1, m2 = st.columns([1, 3])
                        with m1:
                            st.markdown(f"""
                                <div class="score-container" style="border-top-color: {score_color};">
                                    <div class="score-value" style="color: {score_color};">{score}%</div>
                                    <div class="score-label">{rating}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        with m2:
                            st.markdown("#### Summary Analysis")
                            st.write(gap_analysis.explanation)
                            
                        st.markdown("---")
                        
                        # Display skills side-by-side
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.markdown("#### ✅ Match (Skills You Have)")
                            if gap_analysis.existing_skills:
                                for skill in gap_analysis.existing_skills:
                                    st.markdown(f'<span class="badge badge-existing">{skill.name}</span>', unsafe_allow_html=True)
                            else:
                                st.write("None identified.")
                                
                        with sc2:
                            st.markdown("#### ❌ Gap (Skills to Learn)")
                            if gap_analysis.missing_skills:
                                for skill in gap_analysis.missing_skills:
                                    st.markdown(f'<span class="badge badge-missing">{skill.name}</span>', unsafe_allow_html=True)
                            else:
                                st.write("No major gaps identified!")
                                
                        with sc3:
                            st.markdown("#### ✨ Nice-to-Have (Suggested)")
                            if gap_analysis.suggested_skills:
                                for skill in gap_analysis.suggested_skills:
                                    st.markdown(f'<span class="badge badge-suggested">{skill.name}</span>', unsafe_allow_html=True)
                            else:
                                st.write("None identified.")
                                
                        st.markdown("---")
                        
                        # Strengths and Weaknesses
                        st_col, wk_col = st.columns(2)
                        with st_col:
                            st.markdown("#### 💪 Key Strengths")
                            for stg in gap_analysis.strengths:
                                st.markdown(f"- {stg}")
                        with wk_col:
                            st.markdown("#### ⚠️ Key Weaknesses")
                            for wk in gap_analysis.weaknesses:
                                st.markdown(f"- {wk}")
                                
                    # Tab 3: Learning Roadmap & Recommendations
                    with tab_roadmap:
                        st.markdown("### 📅 30-Day Personalized Learning Roadmap")
                        st.write("Follow this structured 4-week timeline to bridge your missing skill gaps.")
                        
                        for item in roadmap.weekly_plan:
                            with st.expander(f"🗓️ **Week {item.week}: {', '.join(item.goals)}**", expanded=True):
                                r_col1, r_col2 = st.columns(2)
                                with r_col1:
                                    st.markdown("**🔍 Topics to Study:**")
                                    for topic in item.topics:
                                        st.markdown(f"- {topic}")
                                with r_col2:
                                    st.markdown("**🛠️ Practice Tasks & Project Milestones:**")
                                    for task in item.practice_tasks:
                                        st.markdown(f"- {task}")
                                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("### 📚 Recommended Courses & Learning Resources")
                        st.write("We retrieved the following resources from our vetted database matching your skills:")
                        
                        if roadmap.recommended_resources:
                            for res in roadmap.recommended_resources:
                                st.markdown(f"""
                                    <div class="resource-card">
                                        <div class="resource-title">
                                            <a href="{res.url}" target="_blank">🔗 {res.title}</a> 
                                            <span style="font-size: 0.8rem; background-color: #e0e7ff; color: #4338ca; padding: 2px 8px; border-radius: 4px; margin-left: 8px;">{res.resource_type}</span>
                                        </div>
                                        <div style="font-size: 0.9rem; color: #4b5563; margin-top: 4px;">{res.description}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No explicit resource links recommendations found matching the criteria.")
                            
                except Exception as e:
                    st.error(f"An unexpected error occurred during execution: {e}")
                    logger.error(f"Execution failed: {e}")
