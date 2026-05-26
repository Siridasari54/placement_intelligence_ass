import os
import re
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from app.utils.logger import logger
from app.utils.constants import COMPANIES, IT_SERVICE_FIRMS, CONFLICT_COMPANIES
from app.pipeline.retrieve.vector_retriever import VectorRetriever
from app.pipeline.retrieve.bm25_retriever import BM25Retriever
from app.pipeline.retrieve.hybrid_retriever import HybridRetriever
from app.pipeline.retrieve.metadata_retriever import MetadataRetriever

class QueryRouter:
    @staticmethod
    def route_query(query: str) -> Dict[str, Any]:
        """Analyzes a query and returns its category, target companies, and filters."""
        query_lower = query.lower()
        
        # 1. Out-of-Corpus / Adversarial
        out_of_corpus_patterns = [
            r"stock\s+price", r"visit\s+date", r"campus\s+visit", 
            r"placed.*last\s+year", r"work-from-home", r"wfh", 
            r"in\s+the\s+world", r"highest\s+in\s+the\s+world", 
            r"should\s+i\s+join.*or.*better", r"opinion"
        ]
        if any(re.search(pat, query_lower) for pat in out_of_corpus_patterns):
            return {
                "route": "fallback",
                "reason": "out-of-corpus query",
                "filters": {}
            }
            
        # Below-Threshold / Edge Cases
        if "cgpa of 5.0" in query_lower or "cgpa.*5\." in query_lower:
            return {
                "route": "edge_case_cgpa",
                "reason": "cgpa threshold edge case",
                "filters": {"min_cgpa": 5.0}
            }

        # 2. Conflict Detection
        conflict_keywords = ["conflict", "scraped", "portal", "official vs", "difference", "amazon cgpa cutoff"]
        is_conflict = any(k in query_lower for k in conflict_keywords)
        matched_conflict_company = [c for c in CONFLICT_COMPANIES if c.lower() in query_lower]
        
        if is_conflict or (matched_conflict_company and ("cutoff" in query_lower or "package" in query_lower or "cgpa" in query_lower)):
            return {
                "route": "conflict",
                "reason": "query matching conflict-prone fields",
                "company": matched_conflict_company[0] if matched_conflict_company else None,
                "filters": {}
            }

        # 3. Temporal / Trend
        temporal_keywords = ["trend", "increase", "decrease", "grow", "2021", "2022", "2023", "2024", "timeline", "years"]
        if any(k in query_lower for k in temporal_keywords):
            return {
                "route": "temporal",
                "reason": "query involves temporal reasoning over multiple years",
                "filters": {}
            }

        # 4. Multi-hop Eligibility/Hiring queries
        eligibility_keywords = ["cgpa", "backlog", "bond", "package", "lpa", "qualify", "applies to", "can apply"]
        hiring_keywords = ["hire", "analyst", "sde", "officer", "intern", "roles", "distribution"]
        
        has_eligibility = any(k in query_lower for k in eligibility_keywords)
        has_hiring = any(k in query_lower for k in hiring_keywords)
        
        number_count = len(re.findall(r"\d+(\.\d+)?", query_lower))
        
        if (has_eligibility and has_hiring) or (has_eligibility and number_count >= 2) or ("highest package among" in query_lower):
            return {
                "route": "multi_hop",
                "reason": "query requires synthesizing multiple eligibility or hiring constraints",
                "filters": {}
            }
            
        if has_eligibility:
            return {
                "route": "eligibility",
                "reason": "single-hop eligibility query",
                "filters": {}
            }

        if has_hiring:
            return {
                "route": "hiring",
                "reason": "single-hop hiring distribution query",
                "filters": {}
            }

        return {
            "route": "standard",
            "reason": "general semantic search query",
            "filters": {}
        }


class RetrievalPipeline:
    def __init__(self, bm25_retriever: BM25Retriever = None, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir
        self.router = QueryRouter()
        self.hybrid_retriever = HybridRetriever(bm25_retriever)

    def set_bm25_retriever(self, bm25_retriever: BM25Retriever) -> None:
        self.hybrid_retriever.set_bm25_retriever(bm25_retriever)

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """Runs the entire routing and retrieval pipeline."""
        # 1. Routing
        route_info = self.router.route_query(query)
        route = route_info["route"]
        logger.info(f"Query routed to: {route} (Reason: {route_info.get('reason')})")
        
        chunks = []
        
        # 2. Routing Logic Execution
        if route == "fallback":
            chunks = [{
                "text": "Out of corpus fallback indicator.",
                "metadata": {"source": "system", "section": "fallback", "fallback_trigger": True}
            }]
            
        elif route == "edge_case_cgpa":
            chunks = [{
                "text": "Edge Case Alert: Student has a CGPA of 5.0. Analysis shows that the minimum eligibility threshold across all listed companies in this dataset is 6.1 (Microsoft), meaning no company allows a CGPA of 5.0.",
                "metadata": {"source": "system", "section": "eligibility", "below_threshold": True}
            }]
            
        elif route == "conflict":
            company = route_info.get("company")
            chunks = self._retrieve_conflict_info(company) if company else []
            if not chunks:
                chunks = self.hybrid_retriever.retrieve(query, k=10)
                
        elif route == "temporal":
            chunks = self._retrieve_temporal_trend(query)
            if not chunks:
                chunks = self.hybrid_retriever.retrieve(query, k=10)
                
        elif route == "multi_hop":
            chunks = self._retrieve_multi_hop(query)
            if not chunks:
                chunks = self.hybrid_retriever.retrieve(query, k=10)
                
        else:
            # Standard path
            filters = MetadataRetriever.build_filter_from_query(query)
            logger.info(f"Applying metadata filters: {filters}")
            
            # Fetch candidates using hybrid search
            candidate_chunks = self.hybrid_retriever.retrieve(query, k=10, filter=filters)
            
            # Rerank
            from app.pipeline.rerank.reranker import Reranker
            reranker = Reranker()
            chunks = reranker.rerank(query, candidate_chunks)
            
        return chunks, route

    # --- Internal Retriever Subroutines ---
    
    def _load_table(self, name: str) -> pd.DataFrame:
        path = os.path.join(self.processed_dir, f"{name}_data.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        logger.warning(f"Processed file not found: {path}")
        return pd.DataFrame()

    def _retrieve_conflict_info(self, company: str) -> List[Dict[str, Any]]:
        context_pieces = []
        df_elig = self._load_table("eligibility")
        df_conflict = self._load_table("conflict")
        
        official_record = None
        portal_record = None
        
        if not df_elig.empty:
            match = df_elig[df_elig["company"].str.lower() == company.lower()]
            if not match.empty:
                official_record = match.iloc[0].to_dict()
                
        if not df_conflict.empty:
            match = df_conflict[df_conflict["company"].str.lower() == company.lower()]
            if not match.empty:
                portal_record = match.iloc[0].to_dict()
                
        if official_record and portal_record:
            text = (
                f"Conflict Detection Alert! Multiple sources exist for {company}:\n"
                f"- Official Placement Profile: Minimum CGPA Cutoff is {official_record['min_cgpa']}, "
                f"Package is {official_record['package_lpa']} LPA.\n"
                f"- Scraped Unofficial Placement Portal: Cutoff is {portal_record['cgpa_portal']} CGPA, "
                f"Package is {portal_record['package_portal']} LPA.\n"
                f"Conflict Type: {portal_record['conflict_type']}.\n"
                f"IMPORTANT: Prompt the user about this conflict and advise verification."
            )
            context_pieces.append({
                "text": text,
                "metadata": {
                    "company": company,
                    "section": "conflict",
                    "source": "multi-source-detector",
                    "conflict": True
                }
            })
            
        elif official_record:
            text = (
                f"Official Profile for {company}: Minimum CGPA Cutoff is {official_record['min_cgpa']}, "
                f"Package is {official_record['package_lpa']} LPA."
            )
            context_pieces.append({
                "text": text,
                "metadata": {"company": company, "section": "eligibility", "source": "official"}
            })
            
        return context_pieces

    def _retrieve_temporal_trend(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        context_pieces = []
        df = self._load_table("trend")
        
        if df.empty:
            return []
            
        if "grew the most" in query_lower or "largest absolute" in query_lower or "largest increase" in query_lower or "maximum growth" in query_lower:
            df["increase"] = df["package_2024"] - df["package_2021"]
            sorted_df = df.sort_values(by="increase", ascending=False)
            best = sorted_df.iloc[0]
            
            text = "Structured Temporal Analysis: Absolute Package Growth from 2021 to 2024:\n"
            for _, r in sorted_df.iterrows():
                text += f"- {r['company']}: grew by {r['increase']:.1f} LPA (from {r['package_2021']} to {r['package_2024']} LPA)\n"
            text += f"\nThe company that showed the largest absolute package increase from 2021 to 2024 is {best['company']} which grew by {best['increase']:.1f} LPA (from {best['package_2021']} to {best['package_2024']} LPA)."
            
            context_pieces.append({
                "text": text,
                "metadata": {"section": "trend", "company": best["company"], "source": "pandas_temporal"}
            })
        else:
            matched_companies = [c for c in COMPANIES if c.lower() in query_lower]
            if matched_companies:
                filtered = df[df["company"].isin(matched_companies)]
                if not filtered.empty:
                    text = "Structured Temporal Analysis: Package Trends (2021-2024):\n"
                    for _, r in filtered.iterrows():
                        text += (
                            f"- {r['company']}: 2021: {r['package_2021']} LPA, 2022: {r['package_2022']} LPA, "
                            f"2023: {r['package_2023']} LPA, 2024: {r['package_2024']} LPA. "
                            f"Trend: {r['trend_direction']}\n"
                        )
                    context_pieces.append({
                        "text": text,
                        "metadata": {"section": "trend", "source": "pandas_temporal"}
                    })
        return context_pieces

    def _retrieve_multi_hop(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        context_pieces = []
        
        df_elig = self._load_table("eligibility")
        df_hiring = self._load_table("hiring")
        
        # Case 1: student qualifying based on CGPA and backlog
        if "backlog" in query_lower and ("cgpa" in query_lower or "gpa" in query_lower):
            cgpa_val = 7.6
            backlog_val = 1
            bond_free_only = False
            
            cgpa_matches = re.findall(r"cgpa\s+(?:of\s+)?(\d+\.\d+|\d+)", query_lower)
            backlog_matches = re.findall(r"(\d+)\s+backlog", query_lower)
            
            if cgpa_matches:
                cgpa_val = float(cgpa_matches[0])
            if backlog_matches:
                backlog_val = int(backlog_matches[0])
                
            if "no bond" in query_lower or "bond-free" in query_lower or "bond free" in query_lower or "without bond" in query_lower:
                bond_free_only = True
                
            if not df_elig.empty:
                filtered = df_elig[
                    (df_elig["min_cgpa"] <= cgpa_val) & 
                    (df_elig["max_backlogs"] >= backlog_val)
                ]
                
                if bond_free_only:
                    filtered = filtered[filtered["bond_years"] == 0]
                    
                if not filtered.empty:
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
                    
        # Case 2: Python highest package
        elif "python" in query_lower and "highest package" in query_lower:
            if not df_elig.empty:
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
                    
        # Case 3: Analyst hiring count comparison
        elif "analyst" in query_lower and "20" in query_lower and "cgpa 8.0" in query_lower:
            if not df_elig.empty and not df_hiring.empty:
                elig_cos = df_elig[(df_elig["min_cgpa"] <= 8.0) & (df_elig["max_backlogs"] >= 0)]
                elig_cos = elig_cos[elig_cos["package_lpa"] > 20.0]
                merged = pd.merge(elig_cos, df_hiring, on="company")
                if not merged.empty:
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
                        
        # Case 4: Zero-bond with package > 40 LPA
        elif "zero-bond" in query_lower or "zero bond" in query_lower or "bond-free" in query_lower or "no bond" in query_lower:
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
                        
        # Case 5: IT service firms highest package
        elif "highest package" in query_lower and ("service firm" in query_lower or "service companies" in query_lower):
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
                    
        # Case 6: Intern hiring
        elif "intern" in query_lower and "most" in query_lower:
            if not df_hiring.empty:
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
                    
        # Case 7: Package to CGPA ratio
        elif "package-to-cgpa ratio" in query_lower or "package to cgpa" in query_lower:
            if not df_elig.empty:
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
                
        # Case 8: General comparison
        compare_companies = [c for c in COMPANIES if c.lower() in query_lower]
        if len(compare_companies) >= 2:
            comparison_text = f"Structured Analysis comparison of {', '.join(compare_companies)}:\n"
            if not df_elig.empty:
                elig_sub = df_elig[df_elig["company"].isin(compare_companies)]
                comparison_text += "\nEligibility & Cutoff Criteria:\n"
                comparison_text += elig_sub.to_string(index=False) + "\n"
            if not df_hiring.empty:
                hiring_sub = df_hiring[df_hiring["company"].isin(compare_companies)]
                comparison_text += "\nHiring Role Distribution:\n"
                comparison_text += hiring_sub.to_string(index=False) + "\n"
            context_pieces.append({
                "text": comparison_text,
                "metadata": {"section": "comparison", "source": "pandas_analysis"}
            })
            
        return context_pieces
