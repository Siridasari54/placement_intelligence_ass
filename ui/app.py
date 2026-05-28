"""Streamlit ChatGPT-style UI for Placement Intelligence Assistant."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from datetime import datetime
import json
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for ChatGPT-style UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .stChatMessage[data-testid="user"] {
        background-color: #1e1e1e;
    }
    
    .stChatMessage[data-testid="assistant"] {
        background-color: #2d2d2d;
    }
    
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: white;
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    
    .source-expander {
        background-color: #2d2d2d;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
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
    session = {
        "id": st.session_state.current_session_id,
        "timestamp": str(datetime.now()),
        "messages": st.session_state.messages
    }
    st.session_state.chat_sessions.append(session)


def load_chat_session(session_id: str):
    """Load a chat session from history."""
    for session in st.session_state.chat_sessions:
        if session["id"] == session_id:
            st.session_state.messages = session["messages"]
            st.session_state.current_session_id = session_id
            break


def export_chat():
    """Export current chat to JSON."""
    return json.dumps({
        "session_id": st.session_state.current_session_id,
        "timestamp": str(datetime.now()),
        "messages": st.session_state.messages
    }, indent=2)


# Sidebar
with st.sidebar:
    st.title("🎓 Placement Assistant")
    
    # New chat button
    if st.button("New Chat", key="sidebar_new_chat_btn", use_container_width=True):
        save_chat_session()
        st.session_state.messages = []
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()
    
    # Chat history - temporarily disabled to fix button ID conflict
    # st.subheader("Chat History")
    # if st.session_state.chat_sessions:
    #     for idx, session in enumerate(st.session_state.chat_sessions[-settings.ui.max_chat_history:]):
    #         button_text = f"Chat {idx + 1}: {session['timestamp'][:19]}"
    #         if st.button(
    #             button_text,
    #             key=f"history_{session['id']}",
    #             use_container_width=True
    #         ):
    #             load_chat_session(session['id'])
    #             st.rerun()
    
    # Export chat - temporarily disabled to fix button ID conflict
    # st.subheader("Export")
    # # if st.button("Export Chat", key="export_btn", use_container_width=True):
    # #     chat_data = export_chat()
    # #     st.text_area("Export Data", chat_data, height=200, key="export_text")
    
    # System stats
    st.subheader("System Stats")
    st.metric("Total Messages", len(st.session_state.messages))
    st.metric("Chat Sessions", len(st.session_state.chat_sessions))


# Main chat interface
st.title(settings.ui.title)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.markdown(source["text"][:200] + "...")
                    st.caption(f"Metadata: {source['metadata']}")
        
        # Display confidence if available
        if "confidence" in message:
            st.caption(f"Confidence: {message['confidence']:.2%}")


# Chat input
if prompt := st.chat_input("Ask about placement data..."):
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
        with st.spinner("Thinking..."):
            try:
                # Import RAG pipeline
                from core.pipeline import RAGPipeline
                from core.di.factories import register_services
                from core.di.container import ServiceContainer
                
                # Initialize container and services
                container = ServiceContainer()
                register_services(container)
                
                # Get pipeline
                pipeline = container.get_service(RAGPipeline)
                
                if pipeline:
                    # Execute query through pipeline
                    result = pipeline.query(prompt)
                    
                    # Check if pipeline returned meaningful results
                    sources = result.get("sources", [])
                    confidence = result.get("confidence", 0.0)
                    response = result.get("answer", "")
                    
                    # Use fallback if no sources, low confidence, or insufficient information
                    use_fallback = (
                        len(sources) == 0 or 
                        confidence < 0.3 or
                        "don't have enough information" in response.lower() or
                        "not enough information" in response.lower()
                    )
                    
                    if use_fallback:
                        # Fallback to data-driven approach using eligibility_data.json
                        import json
                        import os
                        
                        data_file = "data/processed/eligibility_data.json"
                        if os.path.exists(data_file):
                            with open(data_file, 'r') as f:
                                eligibility_data = json.load(f)
                            
                            # Simple query matching
                            query_lower = prompt.lower()
                            companies = [item["company"].lower() for item in eligibility_data]
                            matched_company = None
                            for company in companies:
                                if company in query_lower:
                                    matched_company = company
                                    break
                            
                            if matched_company:
                                company_data = next((item for item in eligibility_data if item["company"].lower() == matched_company), None)
                                if company_data:
                                    response = f"For {company_data['company']}: Min CGPA is {company_data['min_cgpa']}, Package is {company_data['package_lpa']} LPA, Max backlogs allowed is {company_data['max_backlogs']}."
                                    confidence = 0.9
                                else:
                                    response = "Company data not found."
                            else:
                                response = "No company matched in query."
                            sources = []
                        else:
                            # Keep pipeline result if no data file
                            pass
                    
                    st.markdown(response)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "confidence": confidence,
                        "sources": sources[:3]
                    })
                else:
                    # Fallback to data-driven approach
                    import json
                    data_file = "data/processed/eligibility_data.json"
                    if os.path.exists(data_file):
                        with open(data_file, 'r') as f:
                            eligibility_data = json.load(f)
                        
                        # Simple query matching
                        query_lower = prompt.lower()
                        companies = [item["company"].lower() for item in eligibility_data]
                        matched_company = None
                        for company in companies:
                            if company in query_lower:
                                matched_company = company
                                break
                        
                        if matched_company:
                            company_data = next((item for item in eligibility_data if item["company"].lower() == matched_company), None)
                            if company_data:
                                response = f"For {company_data['company']}: Min CGPA is {company_data['min_cgpa']}, Package is {company_data['package_lpa']} LPA, Max backlogs allowed is {company_data['max_backlogs']}."
                            else:
                                response = "Company data not found."
                        else:
                            response = "No company matched in query."
                        
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "confidence": 0.9,
                            "sources": []
                        })
                    else:
                        error_response = "Data file not found. Please ensure eligibility_data.json exists."
                        st.markdown(error_response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_response,
                            "confidence": 0.0,
                            "sources": []
                        })
                
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
