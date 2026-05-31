"""Analytics UI components for Placement Intelligence Assistant."""

import streamlit as st
from typing import Dict, Any, List


def render_system_health(system_metrics: Dict[str, Any], backend_available: bool = False) -> None:
    """Render system health cards.
    
    Args:
        system_metrics: Dictionary of system metrics
        backend_available: Whether backend services are available
    """
    st.subheader("🏥 System Health")
    
    # Get metrics
    total_queries = system_metrics.get("total_queries", 0)
    avg_latency = system_metrics.get("avg_latency", 0.0)
    avg_confidence = system_metrics.get("avg_confidence", 0.0)
    cache_hit_rate = system_metrics.get("cache_hit_rate", 0.0)
    success_rate = system_metrics.get("success_rate", 1.0)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        status = "🟢 Healthy" if avg_confidence > 0.7 and success_rate > 0.8 else "🟡 Degraded"
        st.metric("System Status", status)
    
    with col2:
        st.metric("Total Queries", total_queries)
    
    with col3:
        latency = f"{avg_latency:.2f}ms"
        st.metric("Avg Latency", latency)
    
    with col4:
        cache_rate = f"{cache_hit_rate:.1%}"
        st.metric("Cache Hit Rate", cache_rate)
    
    with col5:
        success = f"{success_rate:.1%}"
        st.metric("Success Rate", success)


def render_chunk_inspector(last_query_trace: Dict[str, Any]) -> None:
    """Render chunk inspection panel with real retrieved chunks only.
    
    Args:
        last_query_trace: Last query trace data
    """
    st.subheader("🔍 Chunk Inspector")
    
    # Get real chunk data
    if last_query_trace and "retrieved_docs" in last_query_trace:
        chunks = last_query_trace["retrieved_docs"]
        
        if not chunks:
            st.info("No chunk data available. Execute a query to see chunk inspection.")
            return
    else:
        st.info("No retrieval data available yet. Execute a query to see chunk inspection.")
        return
    
    # Display chunks
    for i, chunk in enumerate(chunks):
        score = chunk.get('score', 0.0)
        metadata = chunk.get('metadata', {})
        
        with st.expander(f"Chunk {i+1} - Score: {score:.2f}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Content:**")
                st.markdown(chunk.get('text', 'No content')[:300] + "...")
            
            with col2:
                st.markdown("**Metadata:**")
                for key, value in metadata.items():
                    st.markdown(f"- **{key}:** {value}")


def render_query_history(query_history: List[Dict[str, Any]]) -> None:
    """Render query history.
    
    Args:
        query_history: List of query history entries
    """
    st.subheader("📝 Query History")
    
    if not query_history:
        st.info("No query history yet.")
        return
    
    for item in query_history[-5:]:  # Show last 5 queries
        st.markdown(f"""
        <div class="metric-card">
            <p><strong>Query:</strong> {item['query']}</p>
            <p><strong>Time:</strong> {item['timestamp']}</p>
            <p><strong>Confidence:</strong> {item['confidence']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
