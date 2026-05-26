import re
from typing import List, Dict, Any

class TextParser:
    @staticmethod
    def parse_interview_experiences(text: str) -> List[Dict[str, Any]]:
        """Parses the text narrative of interview experiences by company."""
        # Split by company blocks
        # Matches patterns like "TCS | Technical Focus: System Design"
        company_blocks = re.split(r"\n*(?=[A-Za-z0-9\s&]+ \| Technical Focus:)", text)
        
        parsed = []
        for block in company_blocks:
            block = block.strip()
            if not block:
                continue
                
            lines = block.split("\n")
            header = lines[0]
            
            # Extract company and focus
            header_match = re.match(r"([A-Za-z0-9\s&]+) \| Technical Focus:\s*(.*)", header)
            if not header_match:
                continue
                
            company = header_match.group(1).strip()
            tech_focus = header_match.group(2).strip()
            
            # Parse round details and tips
            rounds = []
            tip = ""
            
            current_round_num = None
            current_round_text = ""
            
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                
                # Check for Tips
                if line.startswith("Tip:") or line.startswith("Tip "):
                    tip = line[4:].strip()
                    continue
                
                # Check if it defines a round, e.g. "Round 1 Online Assessment: ..." or "Round 1 Details"
                round_match = re.match(r"(Round \d+)\s*(.*)", line)
                if round_match:
                    if current_round_num:
                        rounds.append({
                            "round": current_round_num,
                            "details": current_round_text.strip()
                        })
                    current_round_num = round_match.group(1).strip()
                    current_round_text = round_match.group(2).strip()
                else:
                    if current_round_num:
                        current_round_text += " " + line
                    else:
                        # Header description or general notes
                        pass
            
            # Append last round
            if current_round_num:
                rounds.append({
                    "round": current_round_num,
                    "details": current_round_text.strip()
                })
                
            parsed.append({
                "company": company,
                "tech_focus": tech_focus,
                "rounds": rounds,
                "tip": tip,
                "section": "interview",
                "source": "official"
            })
            
        return parsed
