import os
from typing import Dict, Any
from app.utils.logger import logger

class ImageLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found at: {file_path}")

    def load(self) -> Dict[str, Any]:
        """Loads image safely, running optional OCR if packages exist, otherwise falling back."""
        extracted_text = ""
        caption = ""
        
        basename = os.path.basename(self.file_path).lower()
        caption = self._get_chart_caption(basename)
        
        try:
            # Wrap pillow, cv2, and pytesseract imports in safe try-except blocks
            from PIL import Image
            import cv2
            
            img = cv2.imread(self.file_path)
            if img is not None:
                try:
                    import pytesseract
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    extracted_text = pytesseract.image_to_string(rgb_img)
                except Exception as ocr_err:
                    logger.warning(f"pytesseract OCR not available: {ocr_err}")
        except ImportError as imp_err:
            logger.warning(f"Image processing libraries (PIL/OpenCV) not installed: {imp_err}")
        except Exception as e:
            logger.error(f"Error loading image {self.file_path}: {e}")
            
        return {
            "text": extracted_text or caption or f"Image file: {basename}",
            "caption": caption,
            "metadata": {
                "source": os.path.basename(self.file_path),
                "path": self.file_path,
                "type": "image",
                "has_caption": bool(caption)
            }
        }

    def _get_chart_caption(self, filename: str) -> str:
        """Simulates a VLM Captioner converting hiring charts to text descriptions."""
        from app.utils.constants import COMPANIES
        
        matched_company = None
        for c in COMPANIES:
            if c.lower().replace(" ", "_") in filename or c.lower() in filename:
                matched_company = c
                break
                
        if not matched_company:
            return ""
            
        hiring_db = {
            "TCS": "SDE: 88, Analyst: 42, Officer: 70, Intern: 44, Total: 244",
            "Infosys": "SDE: 30, Analyst: 68, Officer: 62, Intern: 22, Total: 182",
            "Deloitte": "SDE: 42, Analyst: 85, Officer: 62, Intern: 44, Total: 233",
            "Accenture": "SDE: 25, Analyst: 22, Officer: 52, Intern: 68, Total: 167",
            "Amazon": "SDE: 42, Analyst: 36, Officer: 40, Intern: 82, Total: 200",
            "Flipkart": "SDE: 58, Analyst: 55, Officer: 50, Intern: 32, Total: 195",
            "Google": "SDE: 30, Analyst: 92, Officer: 46, Intern: 30, Total: 198",
            "Microsoft": "SDE: 58, Analyst: 58, Officer: 36, Intern: 68, Total: 220",
            "Wipro": "SDE: 42, Analyst: 92, Officer: 40, Intern: 82, Total: 256",
            "Cognizant": "SDE: 48, Analyst: 28, Officer: 82, Intern: 34, Total: 192",
            "Capgemini": "SDE: 68, Analyst: 38, Officer: 50, Intern: 58, Total: 214",
            "IBM": "SDE: 58, Analyst: 38, Officer: 78, Intern: 68, Total: 242",
            "Adobe": "SDE: 42, Analyst: 80, Officer: 62, Intern: 48, Total: 232",
            "Oracle": "SDE: 35, Analyst: 92, Officer: 62, Intern: 95, Total: 284",
            "SAP": "SDE: 48, Analyst: 42, Officer: 28, Intern: 38, Total: 156",
            "HCL": "SDE: 48, Analyst: 42, Officer: 38, Intern: 32, Total: 160",
            "Tech Mahindra": "SDE: 58, Analyst: 28, Officer: 58, Intern: 30, Total: 174",
            "Qualcomm": "SDE: 25, Analyst: 38, Officer: 82, Intern: 78, Total: 223",
            "Intel": "SDE: 48, Analyst: 48, Officer: 42, Intern: 48, Total: 186",
            "Samsung R&D": "SDE: 42, Analyst: 80, Officer: 42, Intern: 38, Total: 202"
        }
        
        details = hiring_db.get(matched_company, "unknown role distribution")
        return f"Hiring distribution chart for {matched_company} shows roles: {details}."
