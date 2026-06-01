"""Resume Analyzer backend for skill gap analysis and company matching."""

import io
import json
import logging
from typing import Dict, Any, List, Tuple
from groq import Groq
from config.settings import settings

logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    """Parses resumes (PDF, DOCX, TXT) and performs LLM-based analysis against placement requirements."""
    
    def __init__(self):
        self.api_key = settings.groq_api_key
        self.model = settings.generation.model
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Error initializing Groq client in ResumeAnalyzer: {e}")
        logger.info("ResumeAnalyzer initialized")
        
    def extract_text(self, file_bytes: bytes, file_name: str) -> str:
        """Extract plain text from uploaded file bytes based on file extension.
        
        Args:
            file_bytes: Bytes of the uploaded file
            file_name: Name of the file with extension
            
        Returns:
            Extracted plain text
        """
        ext = file_name.split(".")[-1].lower()
        logger.info(f"Extracting text from resume with format: {ext}")
        
        if ext == "pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
            except Exception as e:
                logger.error(f"Error extracting PDF via pypdf: {e}")
                # Try fallback using pdfplumber if available
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        return "\n".join([page.extract_text() or "" for page in pdf.pages])
                except Exception as ex:
                    logger.error(f"Fallback PDF extraction failed: {ex}")
                    raise RuntimeError(f"Could not parse PDF file: {str(e)}")
                    
        elif ext in ["docx", "doc"]:
            try:
                import docx2txt
                # docx2txt requires a file path or file-like object. 
                # We can write to a temp buffer.
                text = docx2txt.process(io.BytesIO(file_bytes))
                return text
            except Exception as e:
                logger.error(f"Error extracting Word doc: {e}")
                raise RuntimeError(f"Could not parse Word document: {str(e)}")
                
        else:
            # Assume TXT/CSV/plain text
            try:
                return file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    return file_bytes.decode("latin1")
                except Exception as e:
                    raise RuntimeError(f"Could not decode text file: {str(e)}")

    def analyze_resume(self, resume_text: str, companies_data_path: str = "data/processed/eligibility_data.json") -> str:
        """Perform semantic skill gap analysis and matching against placement dataset.
        
        Args:
            resume_text: Text of the student's resume
            companies_data_path: Path to the processed company json file
            
        Returns:
            Formatted Markdown analysis report
        """
        if not self.client:
            return "### 📄 Resume Analysis Error\nAI analysis key is missing. Unable to perform resume evaluation."
            
        # Load companies data
        try:
            with open(companies_data_path, "r") as f:
                companies_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading company data for resume analysis: {e}")
            return "### 📄 Resume Analysis Error\nFailed to load company eligibility dataset for matching."
            
        # Standardize company format for prompt
        companies_summary = []
        for c in companies_data:
            companies_summary.append({
                "company": c["company"],
                "min_cgpa": c["min_cgpa"],
                "max_backlogs": c["max_backlogs"],
                "package_lpa": c["package_lpa"],
                "tech_focus": c["tech_focus"],
                "key_topics": c["key_topics"]
            })
            
        prompt = f"""You are a senior technical recruiter and talent advisor for college placements.
Analyze the following student resume text and compare it against the list of recruiting companies.

Student Resume Text:
\"\"\"
{resume_text}
\"\"\"

Recruiting Companies eligibility & tech stack requirements:
{json.dumps(companies_summary, indent=2)}

Please generate an interactive, premium-grade Placement Alignment Report in Markdown.
The report must include:
1. **Executive Profile Summary**: Brief summary of the candidate's core strengths, estimated skill level, and technical profile.
2. **Top Match Companies**: A list of the top 3-4 company matches based on technical skill alignment. For each, show:
   - Match Score (0-100%)
   - Package (LPA)
   - Reason for match
3. **Skill Gap Analysis**: Compare the candidate's resume skills against the company requirements. Explicitly list:
   - **Matched Skills**: Technical keywords present in both the resume and the companies.
   - **Missing Key Skills**: Core technologies or concepts from the company requirements that are missing in the resume (e.g. DSA, System Design, specific languages).
4. **Actionable Roadmap**: 3-4 specific steps the candidate can take to improve their resume (e.g. certs to get, projects to add, mock tests to practice) to qualify for higher-paying companies.

Be encouraging, highly professional, and ensure all recommendations are strictly grounded in our company requirements.

Markdown Report:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior recruiter assisting engineering students with placement preparation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in LLM resume analysis: {e}")
            return f"### 📄 Resume Analysis\nAn error occurred during AI analysis: {str(e)}"
