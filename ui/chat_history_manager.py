"""Chat history management module for persistent chat sessions."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Manages persistent chat history storage and retrieval."""
    
    def __init__(self, storage_path: str = "data/chat_history.json"):
        """Initialize chat history manager.
        
        Args:
            storage_path: Path to chat history JSON file
        """
        self.storage_path = storage_path
        self._ensure_storage_dir()
        self._load_history()
        
    def _ensure_storage_dir(self):
        """Ensure storage directory exists."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
    
    def _load_history(self) -> None:
        """Load chat history from file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"Loaded {len(self.history)} chat sessions from storage")
            except Exception as e:
                logger.error(f"Error loading chat history: {e}")
                self.history = []
        else:
            self.history = []
    
    def _save_history(self) -> None:
        """Save chat history to file."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.history)} chat sessions to storage")
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")
    
    def save_session(self, session_id: str, messages: List[Dict[str, Any]], title: Optional[str] = None) -> None:
        """Save a chat session.
        
        Args:
            session_id: Unique session identifier
            messages: List of chat messages
            title: Optional session title (auto-generated from first query if not provided)
        """
        # Auto-generate title from first user message if not provided
        if not title and messages:
            for msg in messages:
                if msg.get("role") == "user":
                    title = msg.get("content", "")[:50] + ("..." if len(msg.get("content", "")) > 50 else "")
                    break
        
        if not title:
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Check if session already exists and update it
        for idx, session in enumerate(self.history):
            if session["id"] == session_id:
                self.history[idx] = {
                    "id": session_id,
                    "title": title,
                    "timestamp": str(datetime.now()),
                    "messages": messages,
                    "updated": str(datetime.now())
                }
                self._save_history()
                return
        
        # Add new session
        self.history.append({
            "id": session_id,
            "title": title,
            "timestamp": str(datetime.now()),
            "messages": messages,
            "updated": str(datetime.now())
        })
        self._save_history()
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a chat session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        for session in self.history:
            if session["id"] == session_id:
                return session
        return None
    
    def get_all_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all chat sessions, most recent first.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session data
        """
        # Sort by updated timestamp, most recent first
        sorted_sessions = sorted(
            self.history,
            key=lambda x: x.get("updated", x.get("timestamp", "")),
            reverse=True
        )
        return sorted_sessions[:limit]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        for idx, session in enumerate(self.history):
            if session["id"] == session_id:
                del self.history[idx]
                self._save_history()
                return True
        return False
    
    def clear_all(self) -> None:
        """Clear all chat history."""
        self.history = []
        self._save_history()
        logger.info("Cleared all chat history")
    
    def export_to_txt(self, session_id: Optional[str] = None) -> str:
        """Export chat history to text format.
        
        Args:
            session_id: Specific session to export, or None for all sessions
            
        Returns:
            Text content
        """
        lines = []
        lines.append("=" * 60)
        lines.append("PLACEMENT INTELLIGENCE ASSISTANT - CHAT HISTORY")
        lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")
        
        sessions_to_export = []
        if session_id:
            session = self.load_session(session_id)
            if session:
                sessions_to_export = [session]
        else:
            sessions_to_export = self.get_all_sessions()
        
        for session in sessions_to_export:
            lines.append(f"Session: {session.get('title', 'Untitled')}")
            lines.append(f"ID: {session.get('id', 'N/A')}")
            lines.append(f"Created: {session.get('timestamp', 'N/A')}")
            lines.append("-" * 60)
            
            for msg in session.get("messages", []):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                lines.append(f"[{role}]")
                lines.append(content)
                lines.append("")
            
            lines.append("=" * 60)
            lines.append("")
        
        return "\n".join(lines)
