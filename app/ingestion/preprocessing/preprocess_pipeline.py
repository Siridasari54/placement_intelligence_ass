from app.ingestion.preprocessing.clean_text import clean_unicode_characters
from app.ingestion.preprocessing.normalize_text import normalize_whitespace
from app.ingestion.preprocessing.remove_noise import remove_page_numbers_and_headers

class PreprocessPipeline:
    @staticmethod
    def preprocess(text: str) -> str:
        """Executes full cleaning pipeline on raw text."""
        if not text:
            return ""
        
        # Clean special unicode characters
        text = clean_unicode_characters(text)
        
        # Remove header/footer page patterns
        text = remove_page_numbers_and_headers(text)
        
        # Normalize double spacing / whitespace
        text = normalize_whitespace(text)
        
        return text
