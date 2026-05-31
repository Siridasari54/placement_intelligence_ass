"""Streamlit ChatGPT-style UI for Placement Intelligence Assistant with Hallucination Guards and Diagnostics."""

import sys
import os
from datetime import datetime
import json
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for Premium ChatGPT-style UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .stChatMessage[data-testid="user"] {
        background-color: #1e1e1e;
        border-left: 4px solid #4A90E2;
    }
    
    .stChatMessage[data-testid="assistant"] {
        background-color: #2d2d2d;
        border-left: 4px solid #4CAF50;
    }
    
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: white;
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    
    .source-expander {
        background-color: #2d2d2d;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
        margin-right: 0.5rem;
    }
    
    .badge-pass { background-color: #2e7d32; }
    .badge-warn { background-color: #ef6c00; }
    .badge-fail { background-color: #c62828; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


def save_chat_session():
    """Save current chat session to history."""
    if not st.session_state.messages:
        return
        
    # Check if session already exists
    existing_idx = -1
    for i, s in enumerate(st.session_state.chat_sessions):
        if s["id"] == st.session_state.current_session_id:
            existing_idx = i
            break
            
    # Auto-generate title from first user query
    title = "New Conversation"
    first_msg = next((m for m in st.session_state.messages if m["role"] == "user"), None)
    if first_msg:
        title = first_msg["content"][:30] + "..." if len(first_msg["content"]) > 30 else first_msg["content"]
        
    session = {
        "id": st.session_state.current_session_id,
        "title": title,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": st.session_state.messages.copy()
    }
    
    if existing_idx != -1:
        st.session_state.chat_sessions[existing_idx] = session
    else:
        st.session_state.chat_sessions.append(session)


def load_chat_session(session_id: str):
    """Load a chat session from history."""
    save_chat_session()  # Save current first
    for session in st.session_state.chat_sessions:
        if session["id"] == session_id:
            st.session_state.messages = session["messages"].copy()
            st.session_state.current_session_id = session_id
            break


def export_chat() -> str:
    """Export current chat to JSON."""
    return json.dumps({
        "session_id": st.session_state.current_session_id,
        "timestamp": str(datetime.now()),
        "messages": st.session_state.messages
    }, indent=2)


# Sidebar layout
with st.sidebar:
    st.title("🎓 Placement Control Center")
    
    # New chat button
    if st.button("➕ New Chat", key="sidebar_new_chat_btn", use_container_width=True):
        save_chat_session()
        st.session_state.messages = []
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()
    
    # Chat history switching (re-enabled with unique keys)
    st.subheader("🕓 Chat History")
    if st.session_state.chat_sessions:
        for idx, session in enumerate(reversed(st.session_state.chat_sessions[-settings.ui.max_chat_history:])):
            button_text = f"💬 {session['title']}"
            if st.button(
                button_text,
                key=f"history_btn_{session['id']}_{idx}",
                use_container_width=True
            ):
                load_chat_session(session['id'])
                st.rerun()
    else:
        st.caption("No chat history available")
    
    # Export current chat (re-enabled)
    st.subheader("📥 Export & Share")
    if st.session_state.messages:
        chat_data = export_chat()
        st.download_button(
            label="Export Current Chat (JSON)",
            data=chat_data,
            file_name=f"chat_export_{st.session_state.current_session_id}.json",
            mime="application/json",
            use_container_width=True,
            key="export_chat_btn"
        )
    else:
        st.caption("Chat is empty, nothing to export")
        
    # System Stats
    st.subheader("📊 System Stats")
    st.metric("Total Messages", len(st.session_state.messages))
    st.metric("Saved Conversations", len(st.session_state.chat_sessions))


# Main chat interface
st.title(settings.ui.title)
st.caption("Advanced RAG Assistant with System 2 Attention, Self-Consistency, and Recitation Checks.")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 1. Display Reliability metrics if available
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
            
            # Display warnings for warnings/failures
            issues = rel.get("issues", [])
            if issues and verdict != "PASS":
                with st.expander("⚠️ View Guard Warnings"):
                    for issue in issues:
                        st.markdown(f"- {issue}")
        
        # 2. Display sources if available
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Referenced Sources"):
                for i, source in enumerate(msg["sources"]):
                    st.markdown(f"**[Source {i+1}]:**")
                    st.markdown(source["text"][:300] + "...")
                    st.caption(f"Metadata: {source['metadata']}")
        
        # 3. Display confidence if available
        if "confidence" in msg:
            st.caption(f"Retrieval Confidence: {msg['confidence']:.2%}")


# Chat input
if prompt := st.chat_input("Ask about college placement data..."):
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response using RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Executing advanced RAG pipeline stages..."):
            try:
                # Initialize container and RAG pipeline
                from core.pipeline import RAGPipeline
                from core.di.factories import register_services
                from core.di.container import ServiceContainer
                
                container = ServiceContainer()
                register_services(container)
                pipeline = container.get_service(RAGPipeline)
                
                if pipeline:
                    result = pipeline.query(prompt)
                    
                    sources = result.get("sources", [])
                    confidence = result.get("confidence", 0.0)
                    response = result.get("answer", "")
                    reliability = result.get("reliability", {})
                    
                    st.markdown(response)
                    
                    # Store response in session state
                    msg_entry = {
                        "role": "assistant",
                        "content": response,
                        "confidence": confidence,
                        "sources": sources[:3],
                        "reliability": reliability
                    }
                    st.session_state.messages.append(msg_entry)
                    
                    # Auto save conversation
                    save_chat_session()
                    st.rerun()
                else:
                    st.error("RAG Pipeline service could not be retrieved from container.")
            except Exception as e:
                logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
                error_response = f"I apologize, but I encountered an error processing your query: {str(e)}"
                st.markdown(error_response)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_response,
                    "confidence": 0.0,
                    "sources": []
                })
