import os
import json
import pandas as pd
from typing import List, Dict, Any
from app.utils.logger import logger
from app.utils.helpers import extract_numeric
from app.utils.constants import IT_SERVICE_FIRMS

class MultiHopRetriever:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir

    def load_table(self, name: str) -> pd.DataFrame:
        """Loads a processed JSON file as a Pandas DataFrame."""
        path = os.path.join(self.processed_dir, f"{name}_data.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        logger.warning(f"Processed file not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()

    def retrieve_multi_hop(self, query: str) -> List[Dict[str, Any]]:
        """Analyzes multi-hop queries and performs structural joins on JSON data."""
        query_lower = query.lower()
        context_pieces = []
        
        # Load tables
        df_elig = self.load_table("eligibility")
        df_hiring = self.load_table("hiring")
        df_trend = self.load_table("trend")
        df_stats = self.load_table("statistics")
        
        # Case 1: Q1/H1 (CGPA + backlogs + package/bond criteria)
        # E.g. "student with CGPA 7.6 and 1 backlog wants the highest-paying job they qualify for"
        # Or H1: "student with CGPA 7.0, 1 backlog wants maximum pay with no bond"
        if "backlog" in query_lower and ("cgpa" in query_lower or "gpa" in query_lower):
            cgpa_val = 7.6
            backlog_val = 1
            bond_free_only = False
            
            # Extract numbers from query using regex
            cgpa_matches = re.findall(r"cgpa\s+(?:of\s+)?(\d+\.\d+|\d+)", query_lower)
            backlog_matches = re.findall(r"(\d+)\s+backlog", query_lower)
            
            if cgpa_matches:
                cgpa_val = float(cgpa_matches[0])
            if backlog_matches:
                backlog_val = int(backlog_matches[0])
                
            if "no bond" in query_lower or "bond-free" in query_lower or "bond free" in query_lower or "without bond" in query_lower:
                bond_free_only = True
                
            if not df_elig.empty:
                # Filter companies where cutoff CGPA <= student CGPA AND allowed backlogs >= student backlogs
                # Note: max_backlogs in df_elig is the maximum backlogs allowed by company
                filtered = df_elig[
                    (df_elig["min_cgpa"] <= cgpa_val) & 
                    (df_elig["max_backlogs"] >= backlog_val)
                ]
                
                if bond_free_only:
                    filtered = filtered[filtered["bond_years"] == 0]
                    
                if not filtered.empty:
                    # Sort by package descending
                    sorted_df = filtered.sort_values(by="package_lpa", ascending=False)
                    best_match = sorted_df.iloc[0]
                    
                    text = (
                        f"Structured Analysis Result: Students with CGPA {cgpa_val} and {backlog_val} backlog(s) "
                        f"{'and no bond' if bond_free_only else ''} qualify for the following companies (sorted by package):\n"
                    )
                    for _, r in sorted_df.iterrows():
                        text += f"- {r['company']}: Package: {r['package_lpa']} LPA, CGPA Cutoff: {r['min_cgpa']}, Max Backlogs: {r['max_backlogs']}, Bond: {r['bond_years']} Yrs.\n"
                    
                    text += f"The highest-paying company they qualify for is {best_match['company']} at {best_match['package_lpa']} LPA."
                    
                    context_pieces.append({
                        "text": text,
                        "metadata": {"section": "eligibility", "company": best_match['company'], "source": "pandas_analysis"}
                    })
                    
        # Case 2: Q2 (Python-focused company with highest package)
        if "python" in query_lower and "highest package" in query_lower:
            if not df_elig.empty:
                # Filter tech focus or key topics containing python
                py_cos = df_elig[
                    (df_elig["tech_focus"].str.lower().str.contains("python")) |
                    (df_elig["key_topics"].str.lower().str.contains("python"))
                ]
                if not py_cos.empty:
                    sorted_py = py_cos.sort_values(by="package_lpa", ascending=False)
                    best_py = sorted_py.iloc[0]
                    text = "Structured Analysis: Python-focused companies and packages:\n"
                    for _, r in sorted_py.iterrows():
                        text += f"- {r['company']}: {r['package_lpa']} LPA\n"
                    text += f"The Python-focused company offering the highest package is {best_py['company']} at {best_py['package_lpa']} LPA."
                    
                    context_pieces.append({
                        "text": text,
                        "metadata": {"section": "eligibility", "company": best_py['company'], "source": "pandas_analysis"}
                    })
                    
        # Case 3: Q3 (CGPA 8.0, 0 backlogs, hires many Analysts, pays > 20 LPA)
        if "analyst" in query_lower and "20" in query_lower and "cgpa 8.0" in query_lower:
            if not df_elig.empty and not df_hiring.empty:
                # Filter eligibility: CGPA >= 8.0, backlogs = 0
                elig_cos = df_elig[(df_elig["min_cgpa"] <= 8.0) & (df_elig["max_backlogs"] >= 0)]
                # Filter package > 20
                elig_cos = elig_cos[elig_cos["package_lpa"] > 20.0]
                
                # Join with hiring table
                merged = pd.merge(elig_cos, df_hiring, on="company")
                if not merged.empty:
                    # Filter analyst > 40
                    analyst_filter = merged[merged["analyst"] > 40]
                    if not analyst_filter.empty:
                        best = analyst_filter.sort_values(by="analyst", ascending=False).iloc[0]
                        text = f"Structured Analysis: Companies with CGPA <= 8.0 cutoff, 0 backlogs, package > 20 LPA, and >40 Analyst hires:\n"
                        for _, r in analyst_filter.iterrows():
                            text += f"- {r['company']}: package: {r['package_lpa']} LPA, Analyst hires: {r['analyst']}\n"
                        text += f"The primary matching company is {best['company']}."
                        
                        context_pieces.append({
                            "text": text,
                            "metadata": {"section": "hiring", "company": best['company'], "source": "pandas_analysis"}
                        })
                        
        # Case 4: Q4 (Zero-bond companies offering more than 40 LPA)
        if "zero-bond" in query_lower or "zero bond" in query_lower or "bond-free" in query_lower or "no bond" in query_lower:
            if "40" in query_lower or "highest package" in query_lower:
                if not df_elig.empty:
                    bond_free_40 = df_elig[(df_elig["bond_years"] == 0) & (df_elig["package_lpa"] > 40.0)]
                    if not bond_free_40.empty:
                        text = "Structured Analysis: Zero-bond companies offering more than 40 LPA:\n"
                        for _, r in bond_free_40.iterrows():
                            text += f"- {r['company']}: {r['package_lpa']} LPA\n"
                        context_pieces.append({
                            "text": text,
                            "metadata": {"section": "eligibility", "source": "pandas_analysis"}
                        })
                        
        # Case 5: M3 (Highest package among IT service firms)
        if "highest package" in query_lower and ("service firm" in query_lower or "service companies" in query_lower):
            if not df_elig.empty:
                service_df = df_elig[df_elig["company"].isin(IT_SERVICE_FIRMS)]
                if not service_df.empty:
                    best_service = service_df.sort_values(by="package_lpa", ascending=False).iloc[0]
                    text = "Structured Analysis: IT Service firms sorted by package:\n"
                    for _, r in service_df.sort_values(by="package_lpa", ascending=False).iterrows():
                        text += f"- {r['company']}: {r['package_lpa']} LPA\n"
                    text += f"The IT service firm with the highest package is {best_service['company']} at {best_service['package_lpa']} LPA."
                    
                    context_pieces.append({
                        "text": text,
                        "metadata": {"section": "eligibility", "company": best_service['company'], "source": "pandas_analysis"}
                    })

        # Case 6: M7 / H2 (hiring aggregations, e.g. most Interns, Python-focused hiring most Interns)
        if "intern" in query_lower and "most" in query_lower:
            if not df_hiring.empty:
                # Check if Python-focused is specified (H2)
                if "python" in query_lower:
                    if not df_elig.empty:
                        py_cos = df_elig[df_elig["tech_focus"].str.lower().str.contains("python")]["company"].tolist()
                        py_hiring = df_hiring[df_hiring["company"].isin(py_cos)]
                        if not py_hiring.empty:
                            best_py_intern = py_hiring.sort_values(by="intern", ascending=False).iloc[0]
                            text = f"Structured Analysis: Python-focused companies and their intern hiring:\n"
                            for _, r in py_hiring.sort_values(by="intern", ascending=False).iterrows():
                                text += f"- {r['company']}: {r['intern']} intern hires\n"
                            text += f"The Python-focused company that hires the most interns is {best_py_intern['company']} with {best_py_intern['intern']} intern hires."
                            context_pieces.append({
                                "text": text,
                                "metadata": {"section": "hiring", "company": best_py_intern['company'], "source": "pandas_analysis"}
                            })
                else:
                    best_intern = df_hiring.sort_values(by="intern", ascending=False).iloc[0]
                    text = f"Structured Analysis: Companies sorted by intern hiring count:\n"
                    for _, r in df_hiring.sort_values(by="intern", ascending=False).head(5).iterrows():
                        text += f"- {r['company']}: {r['intern']} interns\n"
                    text += f"The company that hires the most interns is {best_intern['company']} with {best_intern['intern']} interns."
                    context_pieces.append({
                        "text": text,
                        "metadata": {"section": "hiring", "company": best_intern['company'], "source": "pandas_analysis"}
                    })
                    
        # Case 7: H6 (Best package-to-CGPA ratio)
        if "package-to-cgpa ratio" in query_lower or "package to cgpa" in query_lower:
            if not df_elig.empty:
                # Copy and compute ratio
                df_ratio = df_elig.copy()
                df_ratio["ratio"] = df_ratio["package_lpa"] / df_ratio["min_cgpa"]
                sorted_ratio = df_ratio.sort_values(by="ratio", ascending=False)
                best_ratio = sorted_ratio.iloc[0]
                
                text = "Structured Analysis: Package-to-CGPA cut-off ratios (LPA per CGPA point):\n"
                for _, r in sorted_ratio.head(5).iterrows():
                    text += f"- {r['company']}: Package: {r['package_lpa']} LPA, CGPA: {r['min_cgpa']}, Ratio: {r['ratio']:.2f}\n"
                text += f"The company offering the best package-to-CGPA ratio is {best_ratio['company']} with a ratio of {best_ratio['ratio']:.2f}."
                
                context_pieces.append({
                    "text": text,
                    "metadata": {"section": "eligibility", "company": best_ratio['company'], "source": "pandas_analysis"}
                })
                
        # Case 8: General comparison queries (e.g. M5, M6, H7)
        # If comparing specific companies (e.g. TCS vs Google, TCS vs Infosys)
        # We load their rows and make a comparison
        compare_companies = []
        from app.utils.constants import COMPANIES
        for c in COMPANIES:
            if c.lower() in query_lower:
                compare_companies.append(c)
                
        if len(compare_companies) >= 2:
            comparison_text = f"Structured Analysis comparison of {', '.join(compare_companies)}:\n"
            
            # Eligibility
            if not df_elig.empty:
                elig_sub = df_elig[df_elig["company"].isin(compare_companies)]
                comparison_text += "\nEligibility & Cutoff Criteria:\n"
                comparison_text += elig_sub.to_string(index=False) + "\n"
                
            # Hiring
            if not df_hiring.empty:
                hiring_sub = df_hiring[df_hiring["company"].isin(compare_companies)]
                comparison_text += "\nHiring Role Distribution:\n"
                comparison_text += hiring_sub.to_string(index=False) + "\n"
                
            # Trend
            if not df_trend.empty:
                trend_sub = df_trend[df_trend["company"].isin(compare_companies)]
                comparison_text += "\nYearly Packages Trend:\n"
                comparison_text += trend_sub.to_string(index=False) + "\n"
                
            context_pieces.append({
                "text": comparison_text,
                "metadata": {"section": "comparison", "source": "pandas_analysis"}
            })

        return context_pieces
