"""CSS styles for Placement Intelligence Assistant UI."""


def get_custom_css() -> str:
    """Return custom CSS for the application."""
    return """
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
    
    .metric-card {
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    
    .metric-card.warning {
        border-left-color: #FF9800;
    }
    
    .metric-card.error {
        border-left-color: #f44336;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        border-radius: 0.5rem;
        padding: 1rem;
    }
</style>
"""
