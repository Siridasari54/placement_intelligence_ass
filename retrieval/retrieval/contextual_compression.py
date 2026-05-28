from typing import List, Dict, Any

class ContextualCompression:
    @staticmethod
    def compress_context(chunks: List[Dict[str, Any]], max_tokens: int = 1500) -> List[Dict[str, Any]]:
        """Prunes retrieved chunks to avoid exceeding token budgets, removing redundant sentences."""
        compressed = []
        token_count = 0
        seen_sentences = set()
        
        for chunk in chunks:
            text = chunk.get("text", "")
            # Simple character approximation for fast estimation
            approx_tokens = len(text) // 4
            
            if token_count + approx_tokens > max_tokens:
                if len(compressed) >= 1: # always keep at least 1 document
                    break
                    
            # Deduplicate sentences within context to prune duplicate listings
            sentences = text.split(". ")
            cleaned_sentences = []
            for s in sentences:
                s_clean = s.strip().lower()
                if s_clean not in seen_sentences:
                    seen_sentences.add(s_clean)
                    cleaned_sentences.append(s)
                    
            cleaned_text = ". ".join(cleaned_sentences).strip()
            if cleaned_text:
                new_chunk = chunk.copy()
                new_chunk["text"] = cleaned_text
                compressed.append(new_chunk)
                token_count += approx_tokens
                
        return compressed
