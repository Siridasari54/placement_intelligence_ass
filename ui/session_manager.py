"""Session management for Placement Intelligence Assistant."""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages chat sessions with persistent storage."""
    
    def __init__(self, chat_history_file: str = "data/chat_history.json"):
        """Initialize session manager.
        
        Args:
            chat_history_file: Path to chat history JSON file
        """
        self.chat_history_file = chat_history_file
        self._ensure_data_directory()
    
    def _ensure_data_directory(self) -> None:
        """Ensure data directory exists."""
        os.makedirs(os.path.dirname(self.chat_history_file), exist_ok=True)
    
    def load_chat_history(self) -> List[Dict[str, Any]]:
        """Load chat history from file.
        
        Returns:
            List of chat sessions
        """
        if os.path.exists(self.chat_history_file):
            try:
                with open(self.chat_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading chat history: {e}")
                return []
        return []
    
    def save_chat_history(self, sessions: List[Dict[str, Any]]) -> None:
        """Save chat history to file.
        
        Args:
            sessions: List of chat sessions to save
        """
        try:
            with open(self.chat_history_file, 'w') as f:
                json.dump(sessions, f, indent=2)
            logger.info(f"Chat history saved. Total sessions: {len(sessions)}")
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")
    
    def create_new_session(self, current_messages: List[Dict[str, Any]], 
                          current_session_id: Optional[str]) -> tuple[str, List[Dict[str, Any]]]:
        """Create a new chat session.
        
        Args:
            current_messages: Current chat messages
            current_session_id: Current session ID
            
        Returns:
            Tuple of (new_session_id, new_messages_list)
        """
        new_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return new_session_id, []
    
    def save_current_session(self, sessions: List[Dict[str, Any]], 
                            session_id: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Save current session to sessions list.
        
        Args:
            sessions: List of all sessions
            session_id: Current session ID
            messages: Current session messages
            
        Returns:
            Updated sessions list
        """
        if not session_id or not messages:
            return sessions
        
        # Find existing session
        existing_session = next(
            (s for s in sessions if s["id"] == session_id),
            None
        )
        
        if existing_session:
            # Update existing session
            existing_session["messages"] = messages.copy()
        else:
            # Create new session with auto-generated title
            title = self._generate_title(messages)
            sessions.append({
                "id": session_id,
                "title": title,
                "timestamp": str(datetime.now()),
                "messages": messages.copy()
            })
        
        return sessions
    
    def _generate_title(self, messages: List[Dict[str, Any]]) -> str:
        """Generate title from first user message.
        
        Args:
            messages: List of messages
            
        Returns:
            Generated title
        """
        first_user_msg = next((m for m in messages if m["role"] == "user"), None)
        if first_user_msg:
            content = first_user_msg["content"]
            return content[:50] + "..." if len(content) > 50 else content
        return "New Chat"
    
    def switch_session(self, sessions: List[Dict[str, Any]], 
                      target_session_id: str, current_session_id: Optional[str],
                      current_messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        """Switch to a different session.
        
        Args:
            sessions: List of all sessions
            target_session_id: Session ID to switch to
            current_session_id: Current session ID
            current_messages: Current session messages
            
        Returns:
            Tuple of (updated_sessions, new_session_id, new_messages)
        """
        # Save current session first
        sessions = self.save_current_session(sessions, current_session_id, current_messages)
        
        # Load target session
        target_session = next(
            (s for s in sessions if s["id"] == target_session_id),
            None
        )
        
        if target_session:
            return sessions, target_session_id, target_session["messages"].copy()
        
        return sessions, current_session_id or "", current_messages
    
    def delete_session(self, sessions: List[Dict[str, Any]], 
                      session_id: str, current_session_id: Optional[str]) -> tuple[List[Dict[str, Any]], Optional[str], List[Dict[str, Any]]]:
        """Delete a session.
        
        Args:
            sessions: List of all sessions
            session_id: Session ID to delete
            current_session_id: Current session ID
            
        Returns:
            Tuple of (updated_sessions, new_current_session_id, new_current_messages)
        """
        sessions = [s for s in sessions if s['id'] != session_id]
        
        # Clear current session if deleting it
        if current_session_id == session_id:
            return sessions, None, []
        
        return sessions, current_session_id, []
    
    def export_session(self, session: Dict[str, Any]) -> str:
        """Export session to JSON string.
        
        Args:
            session: Session to export
            
        Returns:
            JSON string of session
        """
        return json.dumps(session, indent=2)
    
    def get_session_stats(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about sessions.
        
        Args:
            sessions: List of all sessions
            
        Returns:
            Dictionary of statistics
        """
        total_sessions = len(sessions)
        total_messages = sum(len(s["messages"]) for s in sessions)
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "avg_messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0
        }
