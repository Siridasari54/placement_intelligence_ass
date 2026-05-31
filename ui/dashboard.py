"""Placement Intelligence Assistant - Production Dashboard."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from datetime import datetime
from typing import Dict, Any, List
import logging

from ui.styles import get_custom_css
from ui.session_manager import SessionManager
from ui.query_handler import QueryHandler
from ui.analytics_ui import render_system_health, render_chunk_inspector, render_query_history

logger = logging.getLogger(__name__)

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Placement Intelligence Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize session state
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

if "current_chat_session_id" not in st.session_state:
    st.session_state.current_chat_session_id = None

if "current_chat_messages" not in st.session_state:
    st.session_state.current_chat_messages = []

if "last_query_trace" not in st.session_state:
    st.session_state.last_query_trace = None

# Initialize managers
session_manager = SessionManager()
query_handler = QueryHandler()

# Load chat history on startup
st.session_state.chat_sessions = session_manager.load_chat_history()


def render_header():
    """Render dashboard header."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem; color: white;">
        <h1>🎓 Placement Intelligence Assistant</h1>
        <p>Production-Ready RAG System for Placement Data</p>
    </div>
    """, unsafe_allow_html=True)


def render_query_interface():
    """Render query interface with RAG pipeline integration."""
    st.subheader("💬 Query Interface")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter your query:", placeholder="e.g., What is the CGPA requirement for Google?", key="query_input")
    
    with col2:
        mode = st.selectbox("Retrieval Mode", ["Auto", "Semantic Heavy", "Keyword Heavy", "Balanced"], key="retrieval_mode")
    
    if st.button("Execute Query", key="execute_query", use_container_width=True):
        if not query:
            st.warning("Please enter a query")
            return
        
        # Execute query
        with st.spinner("Processing query..."):
            result = query_handler.execute_rag_query(query, mode)
            
            # Update session state
            st.session_state.last_query_trace = {
                "query": query,
                "answer": result["answer"],
                "retrieved_docs": result.get("sources", []),
                "confidence": result["confidence"],
                "latency": result["latency"],
                "query_type": result.get("query_type", "unknown"),
                "retrieval_mode": result.get("retrieval_mode", mode)
            }
            
            # Update metrics
            st.session_state.system_metrics["total_queries"] += 1
            st.session_state.system_metrics["avg_latency"] = result["latency"]
            st.session_state.system_metrics["avg_confidence"] = result["confidence"]
            
            # Add to history
            st.session_state.query_history.append({
                "query": query,
                "timestamp": str(datetime.now()),
                "confidence": result["confidence"],
                "latency": result["latency"]
            })
            
            # Ensure current session exists
            if not st.session_state.current_chat_session_id:
                st.session_state.current_chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Add messages to current session
            st.session_state.current_chat_messages.append({
                "role": "user",
                "content": query
            })
            st.session_state.current_chat_messages.append({
                "role": "assistant",
                "content": result["answer"],
                "confidence": result["confidence"],
                "sources": result.get("sources", [])[:3]
            })
            
            # Save session
            st.session_state.chat_sessions = session_manager.save_current_session(
                st.session_state.chat_sessions,
                st.session_state.current_chat_session_id,
                st.session_state.current_chat_messages
            )
            session_manager.save_chat_history(st.session_state.chat_sessions)
            
            st.success("Query executed successfully!")
            
            # Display answer
            st.markdown("### Generated Answer")
            st.markdown(result["answer"])
            
            # Display analytics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result['confidence'] * 100:.0f}%")
            with col2:
                st.metric("Latency", f"{result['latency']:.0f}ms")
            with col3:
                st.metric("Sources Used", len(result.get("sources", [])))
            
            # Display query type and retrieval mode
            st.markdown(f"**Query Type:** {result.get('query_type', 'unknown')}")
            st.markdown(f"**Retrieval Mode:** {result.get('retrieval_mode', mode)}")
            
            # Display sources
            if result.get("sources"):
                st.markdown("### Sources")
                for i, source in enumerate(result["sources"][:3]):
                    with st.expander(f"Source {i+1}"):
                        st.markdown(source.get("text", "No content")[:200] + "...")
                        st.markdown(f"**Metadata:** {source.get('metadata', {})}")


def render_chat_history():
    """Render chat history with multi-session support."""
    st.subheader("💬 Chat History")
    
    # New Chat button
    if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True):
        # Save current session
        st.session_state.chat_sessions = session_manager.save_current_session(
            st.session_state.chat_sessions,
            st.session_state.current_chat_session_id,
            st.session_state.current_chat_messages
        )
        session_manager.save_chat_history(st.session_state.chat_sessions)
        
        # Create new session
        new_session_id, new_messages = session_manager.create_new_session(
            st.session_state.current_chat_messages,
            st.session_state.current_chat_session_id
        )
        st.session_state.current_chat_session_id = new_session_id
        st.session_state.current_chat_messages = new_messages
        st.rerun()
    
    # List previous sessions
    if st.session_state.chat_sessions:
        st.markdown("### 🕓 Previous Chat Sessions")
        
        for session in reversed(st.session_state.chat_sessions):
            with st.expander(f"📝 {session['title']} - {session['timestamp'][:19]}"):
                st.markdown(f"**Session ID:** {session['id']}")
                st.markdown(f"**Messages:** {len(session['messages'])}")
                
                # Conversation preview
                if session['messages']:
                    st.markdown("**Conversation Preview:**")
                    for msg in session['messages'][:3]:
                        role_icon = "👤" if msg['role'] == 'user' else "🤖"
                        st.markdown(f"{role_icon} **{msg['role'].title()}:** {msg['content'][:100]}...")
                    if len(session['messages']) > 3:
                        st.caption(f"... and {len(session['messages']) - 3} more messages")
                
                # Switch button
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔄 Switch", key=f"history_btn_{session['id']}", use_container_width=True):
                        st.session_state.chat_sessions, new_session_id, new_messages = session_manager.switch_session(
                            st.session_state.chat_sessions,
                            session['id'],
                            st.session_state.current_chat_session_id,
                            st.session_state.current_chat_messages
                        )
                        st.session_state.current_chat_session_id = new_session_id
                        st.session_state.current_chat_messages = new_messages
                        session_manager.save_chat_history(st.session_state.chat_sessions)
                        st.rerun()
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"delete_{session['id']}", use_container_width=True):
                        st.session_state.chat_sessions, new_session_id, new_messages = session_manager.delete_session(
                            st.session_state.chat_sessions,
                            session['id'],
                            st.session_state.current_chat_session_id
                        )
                        st.session_state.current_chat_session_id = new_session_id
                        st.session_state.current_chat_messages = new_messages
                        session_manager.save_chat_history(st.session_state.chat_sessions)
                        st.rerun()
    else:
        st.info("No chat sessions yet. Start a new chat to begin!")
    
    # Current session info
    if st.session_state.current_chat_session_id:
        st.markdown("---")
        st.markdown(f"**Current Session ID:** {st.session_state.current_chat_session_id}")
        st.markdown(f"**Messages in current session:** {len(st.session_state.current_chat_messages)}")


def render_sessions_tab():
    """Render sessions management tab."""
    st.subheader("🧠 Session Management")
    
    # Session stats
    stats = session_manager.get_session_stats(st.session_state.chat_sessions)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sessions", stats["total_sessions"])
    with col2:
        st.metric("Total Messages", stats["total_messages"])
    with col3:
        st.metric("Avg Messages/Session", f"{stats['avg_messages_per_session']:.1f}")
    
    st.markdown("---")
    
    # Export current session
    if st.session_state.current_chat_session_id:
        current_session = next(
            (s for s in st.session_state.chat_sessions if s["id"] == st.session_state.current_chat_session_id),
            None
        )
        if current_session:
            if st.button("📥 Export Current Session", key="export_session", use_container_width=True):
                export_data = session_manager.export_session(current_session)
                st.text_area("Export Data", export_data, height=300, key="export_text_area")
    
    st.markdown("---")
    
    # Session list with switch/delete (without new chat button to avoid duplicate key)
    st.subheader("🕓 Previous Chat Sessions")
    
    if st.session_state.chat_sessions:
        for session in reversed(st.session_state.chat_sessions):
            with st.expander(f"📝 {session['title']} - {session['timestamp'][:19]}"):
                st.markdown(f"**Session ID:** {session['id']}")
                st.markdown(f"**Messages:** {len(session['messages'])}")
                
                # Conversation preview
                if session['messages']:
                    st.markdown("**Conversation Preview:**")
                    for msg in session['messages'][:3]:
                        role_icon = "👤" if msg['role'] == 'user' else "🤖"
                        st.markdown(f"{role_icon} **{msg['role'].title()}:** {msg['content'][:100]}...")
                    if len(session['messages']) > 3:
                        st.caption(f"... and {len(session['messages']) - 3} more messages")
                
                # Switch button
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔄 Switch", key=f"sessions_switch_{session['id']}", use_container_width=True):
                        st.session_state.chat_sessions, new_session_id, new_messages = session_manager.switch_session(
                            st.session_state.chat_sessions,
                            session['id'],
                            st.session_state.current_chat_session_id,
                            st.session_state.current_chat_messages
                        )
                        st.session_state.current_chat_session_id = new_session_id
                        st.session_state.current_chat_messages = new_messages
                        session_manager.save_chat_history(st.session_state.chat_sessions)
                        st.rerun()
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"sessions_delete_{session['id']}", use_container_width=True):
                        st.session_state.chat_sessions, new_session_id, new_messages = session_manager.delete_session(
                            st.session_state.chat_sessions,
                            session['id'],
                            st.session_state.current_chat_session_id
                        )
                        st.session_state.current_chat_session_id = new_session_id
                        st.session_state.current_chat_messages = new_messages
                        session_manager.save_chat_history(st.session_state.chat_sessions)
                        st.rerun()
    else:
        st.info("No chat sessions yet. Start a new chat to begin!")


def render_analytics_tab():
    """Render analytics tab."""
    render_system_health(st.session_state.system_metrics)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        render_chunk_inspector(st.session_state.last_query_trace)
    with col2:
        render_query_history(st.session_state.query_history)


def main():
    """Main dashboard application."""
    render_header()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Sessions", "📊 Analytics"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            render_query_interface()
        with col2:
            render_chat_history()
    
    with tab2:
        render_sessions_tab()
    
    with tab3:
        render_analytics_tab()


if __name__ == "__main__":
    main()
