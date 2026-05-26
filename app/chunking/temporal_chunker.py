from typing import List, Dict, Any

class TemporalChunker:
    @staticmethod
    def chunk_trend_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Converts a trend row to multiple chunks (yearly and unified)."""
        company = row["company"]
        p_2021 = row["package_2021"]
        p_2022 = row["package_2022"]
        p_2023 = row["package_2023"]
        p_2024 = row["package_2024"]
        trend = row["trend_direction"]
        
        chunks = []
        
        # Yearly Chunks
        years_map = {2021: p_2021, 2022: p_2022, 2023: p_2023, 2024: p_2024}
        for year, val in years_map.items():
            chunks.append({
                "text": f"Placement package offered by {company} in the year {year} was {val} LPA.",
                "metadata": {
                    "company": company,
                    "section": "trend",
                    "source": "official",
                    "year": year,
                    "package_lpa": val
                }
            })
            
        # Unified Chunk
        unified_text = (
            f"Placement package trend timeline for {company} (2021-2024): "
            f"Year 2021: {p_2021} LPA, "
            f"Year 2022: {p_2022} LPA, "
            f"Year 2023: {p_2023} LPA, "
            f"Year 2024: {p_2024} LPA. "
            f"The 3-Year Trend is described as {trend}."
        )
        chunks.append({
            "text": unified_text,
            "metadata": {
                "company": company,
                "section": "trend",
                "source": "official",
                "year": 2024, # default to latest
                "package_2021": p_2021,
                "package_2022": p_2022,
                "package_2023": p_2023,
                "package_2024": p_2024
            }
        })
        
        return chunks
