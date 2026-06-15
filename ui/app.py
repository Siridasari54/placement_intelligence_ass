"""Streamlit Unified ChatGPT-style UI and Analytics Dashboard for Placement Intelligence Assistant."""

import sys
import os
from datetime import datetime
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from config.settings import settings
from ui.styles import get_custom_css
from ui.session_manager import SessionManager
from ui.query_handler import QueryHandler
from ui.analytics_ui import render_system_health, render_chunk_inspector, render_query_history
from ui.voice_input import transcribe_audio_via_api

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Placement Intelligence Control Center",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state keys
if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "system_metrics" not in st.session_state:
    st.session_state.system_metrics = {
        "total_queries": 0,
        "avg_latency": 0.0,
        "avg_confidence": 0.0,
        "cache_hit_rate": 0.0,
        "success_rate": 1.0
    }

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_query_trace" not in st.session_state:
    st.session_state.last_query_trace = None

if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""

if "voice_result" not in st.session_state:
    st.session_state.voice_result = None

# Initialize managers
session_manager = SessionManager()
query_handler = QueryHandler()

# Load chat history from persistent storage on startup
st.session_state.chat_sessions = session_manager.load_chat_history()

# Setup current session if not selected
if not st.session_state.current_session_id:
    if st.session_state.chat_sessions:
        most_recent = st.session_state.chat_sessions[-1]
        st.session_state.current_session_id = most_recent["id"]
        st.session_state.messages = most_recent["messages"].copy()
    else:
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        st.session_state.messages = []

# Sidebar Control Center
with st.sidebar:
    st.title("🎓 Control Center")
    
    # 1. Unified Navigation Selection
    page = st.radio("Navigation", [
        "💬 Chat Assistant", 
        "🎤 Voice Input",
        "✅ Eligibility Checker", 
        "📄 Resume Analyzer", 
        "⚖️ Company Compare",
        "📊 System Analytics"
    ], key="navigation_page")
    
    st.markdown("---")
    
    if page == "💬 Chat Assistant":
        st.subheader("Actions")
        # Create a new chat session
        if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True):
            if st.session_state.messages:
                st.session_state.chat_sessions = session_manager.save_current_session(
                    st.session_state.chat_sessions,
                    st.session_state.current_session_id,
                    st.session_state.messages
                )
                session_manager.save_chat_history(st.session_state.chat_sessions)
                
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            st.session_state.messages = []
            st.rerun()
            
        # Chat switching list
        st.subheader("🕓 Conversations")
        if st.session_state.chat_sessions:
            for idx, session in enumerate(reversed(st.session_state.chat_sessions)):
                active_prefix = "👉 " if session["id"] == st.session_state.current_session_id else "💬 "
                button_text = f"{active_prefix}{session['title']}"
                if st.button(
                    button_text,
                    key=f"history_btn_{session['id']}_{idx}",
                    use_container_width=True
                ):
                    st.session_state.chat_sessions, st.session_state.current_session_id, st.session_state.messages = session_manager.switch_session(
                        st.session_state.chat_sessions,
                        session["id"],
                        st.session_state.current_session_id,
                        st.session_state.messages
                    )
                    session_manager.save_chat_history(st.session_state.chat_sessions)
                    st.rerun()
        else:
            st.caption("No chat history available")
            
        st.markdown("---")
        
        # Session exporting and removal
        st.subheader("📥 Export & Share")
        if st.session_state.messages:
            current_session = next(
                (s for s in st.session_state.chat_sessions if s["id"] == st.session_state.current_session_id),
                None
            )
            if not current_session:
                current_session = {
                    "id": st.session_state.current_session_id,
                    "title": "Current Chat",
                    "timestamp": str(datetime.now()),
                    "messages": st.session_state.messages
                }
            chat_data = session_manager.export_session(current_session)
            st.download_button(
                label="Export Current Chat (JSON)",
                data=chat_data,
                file_name=f"chat_export_{st.session_state.current_session_id}.json",
                mime="application/json",
                use_container_width=True,
                key="export_chat_btn"
            )
            
            if st.button("🗑️ Delete Current Chat", key="delete_current_btn", use_container_width=True):
                st.session_state.chat_sessions, new_id, new_msgs = session_manager.delete_session(
                    st.session_state.chat_sessions,
                    st.session_state.current_session_id,
                    st.session_state.current_session_id
                )
                session_manager.save_chat_history(st.session_state.chat_sessions)
                st.session_state.current_session_id = new_id
                st.session_state.messages = new_msgs
                st.rerun()
        else:
            st.caption("Chat is empty, nothing to export")
            
        st.subheader("📊 Session Stats")
        stats = session_manager.get_session_stats(st.session_state.chat_sessions)
        st.metric("Total Saved Chats", stats["total_sessions"])
        st.metric("Total Messages", stats["total_messages"])

# ── MAIN LAYOUT Dispatches ──
if page == "💬 Chat Assistant":
    # Custom headers
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎓 Placement Assistant")
        st.caption("Advanced RAG Assistant with System 2 Attention, Self-Consistency, and Recitation Checks.")
    with col2:
        mode = st.selectbox("Retrieval Mode", ["Auto", "Semantic Heavy", "Keyword Heavy", "Balanced"], key="retrieval_mode")
        
    # Render suggestion cards or display current active messages
    if not st.session_state.messages:
        st.markdown("""
        <div class="hero-container">
            <h1 class="gradient-text">Placement Intelligence Control Center</h1>
            <p class="subtitle">An advanced RAG-powered reasoning system for college placements. Ask natural language questions about eligibility cutoffs, interview processes, hiring distributions, and package trends.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("💡 Suggested Queries")
        col_c1, col_c2 = st.columns(2)
        
        suggestions = [
            ("🔍 Google Eligibility", "What is the CGPA requirement for Google?"),
            ("📊 Compare TCS & Infosys", "Compare TCS and Infosys on all eligibility criteria."),
            ("📝 Microsoft Interview Prep", "What topics should I prepare for a Microsoft interview?"),
            ("🚫 Backlog Policies", "List all companies that allow at least 2 backlogs."),
            ("⚖️ Amazon Cutoff Discrepancy", "Is the Amazon CGPA cutoff 6.4 or 7.0? Explain."),
            ("💰 Highest Compensation", "Which company has the highest package in the dataset?")
        ]
        
        for idx, (label, query_text) in enumerate(suggestions):
            target_col = col_c1 if idx % 2 == 0 else col_c2
            with target_col:
                if st.button(f"**{label}**\n\n{query_text}", key=f"suggest_btn_{idx}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": query_text
                    })
                    st.rerun()
    else:
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Display Reliability metrics
                if "reliability" in msg and msg["reliability"]:
                    rel = msg["reliability"]
                    verdict = rel.get("verdict", "PASS")
                    groundedness = rel.get("groundedness_score", 1.0)
                    consistency = rel.get("consistency_score", 1.0)
                    
                    badge_class = "badge-pass" if verdict == "PASS" else "badge-warn" if verdict == "WARN" else "badge-fail"
                    
                    st.markdown(f"""
                    <div style="margin-top: 0.5rem; margin-bottom: 0.5rem;">
                        <span class="badge {badge_class}">🛡️ Reliability: {verdict}</span>
                        <span style="font-size: 0.85rem; color: #aaa; margin-right: 15px;">Groundedness: **{groundedness:.0%}**</span>
                        <span style="font-size: 0.85rem; color: #aaa;">Self-Consistency: **{consistency:.0%}**</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    issues = rel.get("issues", [])
                    if issues and verdict != "PASS":
                        with st.expander("⚠️ View Guard Warnings"):
                            for issue in issues:
                                st.markdown(f"- {issue}")
                
                # Display referenced sources
                if "sources" in msg and msg["sources"]:
                    with st.expander("🔍 View Referenced Sources"):
                        for i, source in enumerate(msg["sources"]):
                            st.markdown(f"**[Source {i+1}]:**")
                            st.markdown(source["text"][:300] + "...")
                            st.caption(f"Metadata: {source['metadata']}")
                
                # Display retrieval confidence
                if "confidence" in msg:
                    st.caption(f"Retrieval Confidence: {msg['confidence']:.2%}")

    # Check if the last message in history is a user prompt (requires answer generation)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        latest_user_message = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Executing advanced RAG pipeline stages..."):
                try:
                    result = query_handler.execute_rag_query(latest_user_message, mode)
                    
                    # Cache last trace
                    st.session_state.last_query_trace = {
                        "query": latest_user_message,
                        "answer": result["answer"],
                        "retrieved_docs": result.get("sources", []),
                        "confidence": result["confidence"],
                        "latency": result["latency"],
                        "query_type": result.get("query_type", "unknown"),
                        "retrieval_mode": result.get("retrieval_mode", mode)
                    }
                    
                    # Track metrics
                    st.session_state.system_metrics["total_queries"] += 1
                    n = st.session_state.system_metrics["total_queries"]
                    if n == 1:
                        st.session_state.system_metrics["avg_latency"] = result["latency"]
                        st.session_state.system_metrics["avg_confidence"] = result["confidence"]
                    else:
                        st.session_state.system_metrics["avg_latency"] = (st.session_state.system_metrics["avg_latency"] * 0.9) + (result["latency"] * 0.1)
                        st.session_state.system_metrics["avg_confidence"] = (st.session_state.system_metrics["avg_confidence"] * 0.9) + (result["confidence"] * 0.1)
                    
                    # Track query history
                    st.session_state.query_history.append({
                        "query": latest_user_message,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "confidence": result["confidence"],
                        "latency": result["latency"]
                    })
                    
                    # Add response to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "confidence": result["confidence"],
                        "sources": result.get("sources", [])[:3],
                        "reliability": result.get("reliability", {})
                    })
                    
                    # Save current session
                    st.session_state.chat_sessions = session_manager.save_current_session(
                        st.session_state.chat_sessions,
                        st.session_state.current_session_id,
                        st.session_state.messages
                    )
                    session_manager.save_chat_history(st.session_state.chat_sessions)
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error executing unified RAG assistant: {e}", exc_info=True)
                    error_msg = f"I apologize, but I encountered an error processing your query: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "confidence": 0.0,
                        "sources": []
                    })
                    st.session_state.chat_sessions = session_manager.save_current_session(
                        st.session_state.chat_sessions,
                        st.session_state.current_session_id,
                        st.session_state.messages
                    )
                    session_manager.save_chat_history(st.session_state.chat_sessions)
                    st.rerun()

    # Chat Input block
    if prompt := st.chat_input("Ask about college placement data..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        st.rerun()

elif page == "🎤 Voice Input":
    st.title("🎤 Voice-Activated Query Assistant")
    st.caption("Ask questions about college placement data using your voice. Transcription is powered by Groq Whisper API.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎙️ Capture Audio")
        voice_lang = st.selectbox("Speech Language", ["English", "Telugu", "Hindi"], index=0, key="voice_page_lang")
        voice_mode = st.selectbox("Retrieval Mode", ["Auto", "Semantic Heavy", "Keyword Heavy", "Balanced"], index=0, key="voice_page_mode")
        
        audio_file = st.audio_input("Record your question:", key="voice_page_audio")
        uploaded_audio = st.file_uploader("Or upload an audio file (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a", "webm", "ogg"], key="voice_page_upload")
        
        selected_audio = audio_file if audio_file else uploaded_audio
        
        transcribe_btn = st.button("Transcribe & Ask Assistant", type="primary", use_container_width=True, disabled=(selected_audio is None))
        
        if selected_audio and transcribe_btn:
            with st.spinner("Transcribing audio..."):
                if hasattr(selected_audio, "seek"):
                    selected_audio.seek(0)
                normalized_query, raw_transcript = transcribe_audio_via_api(selected_audio, voice_lang)
                
            if normalized_query:
                # Store both normalized and raw transcript for debugging
                st.session_state.voice_transcript = raw_transcript
                with st.spinner("Running placement assistant query..."):
                    try:
                        result = query_handler.execute_rag_query(normalized_query, voice_mode)
                        st.session_state.voice_result = result
                        
                        # Cache last trace with both normalized and raw transcript
                        st.session_state.last_query_trace = {
                            "query": normalized_query,
                            "raw_transcript": raw_transcript,
                            "answer": result["answer"],
                            "retrieved_docs": result.get("sources", []),
                            "confidence": result["confidence"],
                            "latency": result["latency"],
                            "query_type": result.get("query_type", "unknown"),
                            "retrieval_mode": result.get("retrieval_mode", voice_mode)
                        }
                        
                        # Track metrics
                        st.session_state.system_metrics["total_queries"] += 1
                        n = st.session_state.system_metrics["total_queries"]
                        if n == 1:
                            st.session_state.system_metrics["avg_latency"] = result["latency"]
                            st.session_state.system_metrics["avg_confidence"] = result["confidence"]
                        else:
                            st.session_state.system_metrics["avg_latency"] = (st.session_state.system_metrics["avg_latency"] * 0.9) + (result["latency"] * 0.1)
                            st.session_state.system_metrics["avg_confidence"] = (st.session_state.system_metrics["avg_confidence"] * 0.9) + (result["confidence"] * 0.1)
                        
                        # Track query history with normalized query
                        st.session_state.query_history.append({
                            "query": normalized_query,
                            "raw_transcript": raw_transcript,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "confidence": result["confidence"],
                            "latency": result["latency"]
                        })
                    except Exception as e:
                        logger.error(f"Error executing voice RAG query: {e}", exc_info=True)
                        st.error(f"An error occurred: {str(e)}")
                        st.session_state.voice_result = None
            else:
                st.error("Audio transcription failed. Please check your voice settings or Groq API key.")
                st.session_state.voice_result = None
                st.session_state.voice_transcript = ""
                
    with col2:
        st.subheader("💬 Response & Insights")
        
        if st.session_state.voice_transcript:
            st.info(f"🗣️ **Transcript:** {st.session_state.voice_transcript}")
            
        if st.session_state.voice_result:
            result = st.session_state.voice_result
            st.markdown(f"### 🎓 Assistant Response\n\n{result['answer']}")
            
            # Confidence & reliability metrics
            if "reliability" in result and result["reliability"]:
                rel = result["reliability"]
                verdict = rel.get("verdict", "PASS")
                groundedness = rel.get("groundedness_score", 1.0)
                consistency = rel.get("consistency_score", 1.0)
                
                badge_class = "badge-pass" if verdict == "PASS" else "badge-warn" if verdict == "WARN" else "badge-fail"
                st.markdown(f"""
                <div style="margin-top: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="badge {badge_class}">🛡️ Reliability: {verdict}</span>
                    <span style="font-size: 0.85rem; color: #aaa; margin-right: 15px;">Groundedness: **{groundedness:.0%}**</span>
                    <span style="font-size: 0.85rem; color: #aaa;">Self-Consistency: **{consistency:.0%}**</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Citations
            if result.get("sources"):
                with st.expander("🔍 View Referenced Sources"):
                    for i, source in enumerate(result["sources"][:3]):
                        st.markdown(f"**[Source {i+1}]:**")
                        st.markdown(source["text"][:300] + "...")
                        st.caption(f"Metadata: {source['metadata']}")
                        
            # Add continue in chat assistant button
            if st.button("💬 Continue in Chat Assistant", key="continue_to_chat_btn", use_container_width=True):
                # Append to active session messages
                st.session_state.messages.append({
                    "role": "user",
                    "content": st.session_state.voice_transcript
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "confidence": result["confidence"],
                    "sources": result.get("sources", [])[:3],
                    "reliability": result.get("reliability", {})
                })
                
                # Save current session
                st.session_state.chat_sessions = session_manager.save_current_session(
                    st.session_state.chat_sessions,
                    st.session_state.current_session_id,
                    st.session_state.messages
                )
                session_manager.save_chat_history(st.session_state.chat_sessions)
                
                # Clear voice page state
                st.session_state.voice_transcript = ""
                st.session_state.voice_result = None
                
                # Switch page to Chat Assistant
                st.session_state.navigation_page = "💬 Chat Assistant"
                st.rerun()
        else:
            if not transcribe_btn:
                st.info("Record or upload an audio query and press 'Transcribe & Ask Assistant'.")

elif page == "✅ Eligibility Checker":
    st.title("✅ Placement Eligibility Checker")
    st.caption("Enter your academic parameters to see which companies you qualify for in this placement cycle.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Your Academic Standing")
        cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        backlogs = st.number_input("Active Backlogs", min_value=0, max_value=10, value=0, step=1)
        branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"])
        check_btn = st.button("Check Eligibility", use_container_width=True)
        
    with col2:
        if check_btn or "eligibility_checked" in st.session_state:
            st.session_state.eligibility_checked = True
            
            # Load eligibility data
            import json
            try:
                with open("data/processed/eligibility_data.json", "r") as f:
                    companies = json.load(f)
            except Exception as e:
                st.error(f"Error loading company data: {e}")
                companies = []
            
            eligible = []
            ineligible = []
            
            for c in companies:
                reasons = []
                if cgpa < c["min_cgpa"]:
                    reasons.append(f"CGPA {cgpa} < Cutoff {c['min_cgpa']}")
                if backlogs > c["max_backlogs"]:
                    reasons.append(f"Backlogs {backlogs} > Allowed {c['max_backlogs']}")
                    
                if not reasons:
                    eligible.append(c)
                else:
                    ineligible.append((c, reasons))
                    
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.subheader(f"🟢 Eligible Companies ({len(eligible)})")
                if eligible:
                    for c in eligible:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #2e7d32; margin-bottom: 12px; background: #132213; border-radius: 8px; padding: 12px;">
                            <h4 style="margin:0 0 5px 0;">{c['company']}</h4>
                            <b>Package:</b> {c['package_lpa']} LPA <br/>
                            <b>Bond:</b> {c['bond_years']} Years <br/>
                            <b>Tech Focus:</b> {c['tech_focus']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("You do not meet the minimum criteria for any companies in this cohort.")
                    
            with col_e2:
                st.subheader(f"🔴 Ineligible Companies ({len(ineligible)})")
                if ineligible:
                    for c, reasons in ineligible:
                        reason_str = ", ".join(reasons)
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #d32f2f; margin-bottom: 12px; background: #261616; border-radius: 8px; padding: 12px;">
                            <h4 style="margin:0 0 5px 0;">{c['company']}</h4>
                            <b>Package:</b> {c['package_lpa']} LPA <br/>
                            <span style="color: #ff8a80;"><b>Reason:</b> {reason_str}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("Congratulations! You are eligible for all companies.")

elif page == "📄 Resume Analyzer":
    st.title("📄 AI Resume Match Analyzer")
    st.caption("Upload your resume to evaluate skill gaps and get tailored recommendations against the recruiting company profiles.")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        
        if st.button("Analyze Resume", use_container_width=True):
            with st.spinner("Extracting text and performing talent gap analysis..."):
                from core.tools.resume_analyzer import ResumeAnalyzer
                analyzer = ResumeAnalyzer()
                try:
                    resume_text = analyzer.extract_text(file_bytes, file_name)
                    report = analyzer.analyze_resume(resume_text)
                    st.success("Resume analysis completed successfully!")
                    st.markdown(report)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

elif page == "⚖️ Company Compare":
    st.title("⚖️ Side-by-Side Company Comparison")
    st.caption("Select two or more companies to compare package packages, cutoffs, bond periods, and technical skills side-by-side.")
    
    import json
    try:
        with open("data/processed/eligibility_data.json", "r") as f:
            companies_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading company data: {e}")
        companies_data = []
        
    company_names = sorted([c["company"] for c in companies_data])
    selected = st.multiselect("Choose Companies to Compare", company_names, default=company_names[:2] if len(company_names) >= 2 else [])
    
    if len(selected) >= 2:
        comparison_list = [c for c in companies_data if c["company"] in selected]
        
        from core.tools.opinion_guard import OpinionGuard
        guard = OpinionGuard()
        report = guard._generate_neutral_comparison(comparison_list)
        st.markdown(report)
    else:
        st.info("Please select at least two companies to generate a comparison.")

elif page == "📊 System Analytics":
    st.title("📊 System Observability & Diagnostics")
    st.caption("Inspect RAG pipeline metrics, retrieved chunks, and query latency distributions.")
    
    render_system_health(st.session_state.system_metrics)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        render_chunk_inspector(st.session_state.last_query_trace)
    with col2:
        render_query_history(st.session_state.query_history)
