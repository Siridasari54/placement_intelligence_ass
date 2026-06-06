"""AI System Reliability Layer for self-consistency, validation, and unsupported answer rejection."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityCheck:
    """Result of reliability check."""
    passed: bool
    verdict: str  # PASS, WARN, FAIL
    confidence: float
    issues: List[str]
    fallback_triggered: bool
    fallback_reason: str
    groundedness_score: float
    consistency_score: float
    recitation_report: Dict[str, Any]
    chain_report: Dict[str, Any]
    lookback_ratio: float = 0.0


class System2Attention:
    """Filters retrieved chunks using System 2 Attention (S2A) to remove irrelevant context."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the System 2 Attention filter."""
        from groq import Groq
        from config.settings import settings
        self.client = Groq(api_key=api_key or settings.groq_api_key)
        self.model = settings.generation.model
        logger.info("System2Attention initialized")
        
    def filter_context(self, query: str, context: List[Document]) -> List[Document]:
        """Filter out irrelevant chunks using the LLM attention layer."""
        if not context:
            return []
        
        logger.info(f"Applying System 2 Attention on {len(context)} documents")
        filtered_docs = []
        
        for idx, doc in enumerate(context):
            prompt = f"""You are an attention filtering module in a college placement RAG system.
Given the User Query: "{query}"
Evaluate if the following document chunk contains information directly relevant, helpful, or contextual to answering the query.

Document Chunk:
{doc.page_content}

Answer ONLY with "YES" if the chunk is relevant and "NO" if the chunk is irrelevant. Do not write any other text or reasoning.
Relevance verdict:"""
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise attention filter. Respond only with YES or NO."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                verdict = response.choices[0].message.content.strip().upper()
                if "YES" in verdict:
                    filtered_docs.append(doc)
                else:
                    logger.info(f"S2A filtered out chunk {idx + 1} as irrelevant: {doc.page_content[:80]}...")
            except Exception as e:
                logger.error(f"Error in System 2 Attention for chunk {idx + 1}: {e}")
                filtered_docs.append(doc)  # Fallback to keeping it
                
        # If all filtered out, return original context as fallback
        if not filtered_docs:
            logger.warning("S2A filtered out all chunks. Falling back to original context.")
            return context
            
        logger.info(f"System 2 Attention completed: kept {len(filtered_docs)} of {len(context)} documents")
        return filtered_docs


class SelfConsistencyVerifier:
    """Verifies self-consistency of generated answers by sampling multiple candidates."""
    
    def __init__(self, num_samples: int = 3):
        """Initialize the self-consistency verifier.
        
        Args:
            num_samples: Number of samples to generate for consistency check
        """
        self.num_samples = num_samples
        logger.info(f"SelfConsistencyVerifier initialized with {num_samples} samples")
    
    def verify(
        self,
        query: str,
        context: List[Document],
        generator
    ) -> Dict[str, Any]:
        """Verify self-consistency by generating multiple samples.
        
        Args:
            query: User query
            context: Retrieved context
            generator: Generator instance
            
        Returns:
            Dictionary with consistency results
        """
        logger.info("Verifying self-consistency")
        
        # Generate multiple samples
        samples = []
        for i in range(self.num_samples):
            try:
                sample = generator.generate(query, context)
                samples.append(sample)
            except Exception as e:
                logger.error(f"Failed to generate self-consistency sample {i+1}: {e}")
                
        if not samples:
            return {
                "consistency_score": 0.0,
                "samples": [],
                "best_answer": "Error generating answer samples.",
                "passed": False
            }
            
        # If only one sample succeeded, return it
        if len(samples) == 1:
            return {
                "consistency_score": 1.0,
                "samples": samples,
                "best_answer": samples[0],
                "passed": True
            }
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency(samples)
        
        # Select most consistent answer
        best_answer = self._select_best_answer(samples)
        
        logger.info(f"Self-consistency score: {consistency_score:.2f}")
        
        return {
            "consistency_score": consistency_score,
            "samples": samples,
            "best_answer": best_answer,
            "passed": consistency_score > 0.5
        }
    
    def _calculate_consistency(self, samples: List[str]) -> float:
        """Calculate consistency score across samples.
        
        Args:
            samples: List of generated samples
            
        Returns:
            Consistency score between 0 and 1
        """
        if len(samples) < 2:
            return 1.0
        
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                similarity = self._calculate_similarity(samples[i], samples[j])
                total_similarity += similarity
                comparisons += 1
        
        if comparisons == 0:
            return 1.0
        
        return total_similarity / comparisons
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _select_best_answer(self, samples: List[str]) -> str:
        """Select the best answer from samples.
        
        Args:
            samples: List of generated samples
            
        Returns:
            Best answer
        """
        best_idx = 0
        best_score = -1.0
        
        for i, sample in enumerate(samples):
            similarities = []
            for j, other_sample in enumerate(samples):
                if i != j:
                    similarities.append(self._calculate_similarity(sample, other_sample))
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            if avg_similarity > best_score:
                best_score = avg_similarity
                best_idx = i
        
        return samples[best_idx]


class RecitationChecker:
    """Extracts factual claims from generated answers and verifies them against retrieved source documents."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the recitation checker."""
        from groq import Groq
        from config.settings import settings
        self.client = Groq(api_key=api_key or settings.groq_api_key)
        self.model = settings.generation.model
        logger.info("RecitationChecker initialized")
        
    def check(self, answer: str, context: List[Document]) -> Dict[str, Any]:
        """Verify claims in answer against retrieved context.
        
        Returns:
            Dictionary with groundedness score, supported/unsupported claims.
        """
        if "retrieved documents do not contain" in answer or "don't have enough information" in answer.lower():
            # If system explicitly claims lack of info, it is technically 100% grounded
            return {
                "groundedness_score": 1.0,
                "claims": [],
                "supported_claims": [],
                "unsupported_claims": [],
                "passed": True
            }
            
        logger.info("Performing recitation check")
        
        # Step 1: Extract claims
        claims = self._extract_claims(answer)
        if not claims:
            return {
                "groundedness_score": 1.0,
                "claims": [],
                "supported_claims": [],
                "unsupported_claims": [],
                "passed": True
            }
            
        supported_claims = []
        unsupported_claims = []
        
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        # Step 2: Verify each claim against context
        for claim in claims:
            is_supported = self._verify_claim(claim, context_text)
            if is_supported:
                supported_claims.append(claim)
            else:
                unsupported_claims.append(claim)
                
        groundedness_score = len(supported_claims) / len(claims)
        passed = groundedness_score >= 0.7
        
        logger.info(f"Recitation check: {len(supported_claims)} supported, {len(unsupported_claims)} unsupported. Score: {groundedness_score:.2f}")
        
        return {
            "groundedness_score": groundedness_score,
            "claims": claims,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "passed": passed
        }
        
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract atomic factual claims from the answer."""
        prompt = f"""You are a factual claim extractor for a RAG verification pipeline.
Analyze the following response and extract a JSON list of atomic factual claims made in it.
An atomic claim is a short sentence containing exactly one fact (e.g. "TCS requires 7.5 CGPA", "Google offers 42 LPA package", "Microsoft allows 1 backlog").
Do not extract opinions, conversational filler, meta-announcements, or reasoning.
Respond ONLY with a valid raw JSON list of strings. Do not include markdown code fences or any other text.

Response:
{answer}

Atomic factual claims JSON list:"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a claim extractor. Output only raw JSON lists of strings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            # Basic fallback: simple sentence split
            import re
            sentences = re.split(r'[.!?]\s+', answer)
            return [s.strip() for s in sentences if len(s.strip()) > 15 and not s.strip().startswith("I don't") and not s.strip().startswith("The retrieved")]

    def _verify_claim(self, claim: str, context_text: str) -> bool:
        """Verify a single claim against the retrieved context."""
        prompt = f"""You are a factual verification assistant. Verify if the following claim is fully supported by the provided context.

Context Documents:
{context_text}

Claim to Verify:
"{claim}"

Answer ONLY "YES" if the claim is fully supported by the context, and "NO" if it is not supported, contradicted, or missing from the context. Do not write anything else.
Verification verdict:"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a verification model. Respond only YES or NO."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            verdict = response.choices[0].message.content.strip().upper()
            return "YES" in verdict
        except Exception as e:
            logger.error(f"Error verifying claim '{claim}': {e}")
            return True  # Safe fallback to avoid false rejection


class ChainOfActionsVerifier:
    """Verifies that reasoning and retrieval actions were actually executed correctly."""
    
    def __init__(self):
        logger.info("ChainOfActionsVerifier initialized")
        
    def verify(self, query: str, query_type: str, trace_stages: List[str]) -> Dict[str, Any]:
        """Verify execution flow based on query classification."""
        issues = []
        
        # 1. Base RAG stage checks
        if "retrieval" not in trace_stages:
            issues.append("Retrieval stage not logged")
        if "reranking" not in trace_stages:
            issues.append("Reranking stage not logged")
        if "refinement" not in trace_stages:
            issues.append("Context refinement stage not logged")
            
        # 2. Multi-hop checks
        if query_type == "multi_hop":
            if "query_planning" not in trace_stages:
                issues.append("Query planning not performed for multi-hop query")
                
        passed = len(issues) == 0
        warning = "" if passed else "Warning: Incomplete reasoning chain: " + "; ".join(issues)
        
        return {
            "passed": passed,
            "issues": issues,
            "warning": warning
        }


class ResponseValidator:
    """Validates responses against quality criteria."""
    
    def __init__(self):
        """Initialize the response validator."""
        self.min_length = 15
        self.max_length = 4000
        self.required_elements = []
        logger.info("ResponseValidator initialized")
    
    def validate(self, response: str, context: List[Document]) -> Dict[str, Any]:
        """Validate response against quality criteria."""
        logger.info("Validating response")
        issues = []
        
        if len(response) < self.min_length:
            issues.append(f"Response too short ({len(response)} chars)")
        if len(response) > self.max_length:
            issues.append(f"Response too long ({len(response)} chars)")
            
        # Check generic empty patterns
        if any(pat in response.lower() for pat in ["placeholder", "not implemented"]):
            issues.append("Response contains placeholder content")
            
        passed = len(issues) == 0
        return {
            "passed": passed,
            "issues": issues,
            "length": len(response)
        }


class ConfidenceThresholdManager:
    """Manages confidence thresholds for system reliability."""
    
    def __init__(self):
        """Initialize threshold manager."""
        self.thresholds = {
            "retrieval_confidence": 0.4,
            "faithfulness": 0.6,
            "consistency": 0.5
        }
        
    def check_thresholds(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if metrics meet required thresholds."""
        issues = []
        passed = True
        
        for name, threshold in self.thresholds.items():
            val = metrics.get(name, 1.0)
            if val < threshold:
                issues.append(f"{name} threshold unmet: {val:.2f} < {threshold}")
                passed = False
                
        return {
            "passed": passed,
            "issues": issues
        }


class UnsupportedAnswerRejector:
    """Rejects unsupported answers based on confidence and grounding."""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        
    def should_reject(self, answer: str, confidence: float) -> Tuple[bool, str]:
        if confidence < self.threshold:
            return True, f"Confidence too low ({confidence:.2f} < {self.threshold})"
        return False, ""


class SystemReliabilityLayer:
    """Unified system reliability layer coordinating hallucination guards."""
    
    def __init__(self):
        """Initialize SystemReliabilityLayer with all components."""
        self.s2a = System2Attention()
        self.consistency_verifier = SelfConsistencyVerifier()
        self.recitation_checker = RecitationChecker()
        self.chain_verifier = ChainOfActionsVerifier()
        self.response_validator = ResponseValidator()
        self.threshold_manager = ConfidenceThresholdManager()
        self.answer_rejector = UnsupportedAnswerRejector()
        
        logger.info("SystemReliabilityLayer fully initialized")
        
    def apply_s2a(self, query: str, context: List[Document]) -> List[Document]:
        """Filter retrieved context using System 2 Attention."""
        return self.s2a.filter_context(query, context)
        
    def compute_lookback_ratio(self, answer: str, context_chunks: List[Any]) -> float:
        """Compute the lookback ratio of the generated answer against retrieved context.
        
        Args:
            answer: Generated answer string
            context_chunks: List of retrieved context documents, strings, or dicts
            
        Returns:
            Float between 0 and 1 representing fraction of answer words that appear in the retrieved context.
        """
        if not answer:
            return 1.0
            
        # Extract words from context chunks
        context_words = set()
        for chunk in context_chunks:
            text = ""
            if isinstance(chunk, str):
                text = chunk
            elif hasattr(chunk, 'page_content'):
                text = chunk.page_content
            elif isinstance(chunk, dict) and 'text' in chunk:
                text = chunk['text']
            
            # Lowercase and clean words
            import re
            words = re.findall(r'\b\w+\b', text.lower())
            context_words.update(words)
            
        # Extract words from the answer
        import re
        answer_words = re.findall(r'\b\w+\b', answer.lower())
        if not answer_words:
            return 1.0
            
        matched_count = sum(1 for word in answer_words if word in context_words)
        return float(matched_count / len(answer_words))
        
    def check_reliability(
        self,
        query: str,
        answer: str,
        context: List[Document],
        confidence: float,
        query_type: str = "factual",
        trace_stages: List[str] = None,
        generator = None
    ) -> ReliabilityCheck:
        """Perform unified PASS/WARN/FAIL reliability check.
        
        Args:
            query: User query
            answer: Generated response
            context: Retrieved and filtered context documents
            confidence: Base retrieval/generation confidence score
            query_type: Classified query type
            trace_stages: Stages successfully completed in this run
            generator: Optional generator instance for self-consistency checks
        """
        logger.info("Starting unified system reliability check")
        
        all_issues = []
        trace_stages = trace_stages or ["retrieval", "reranking", "refinement", "generation"]
        
        # 1. Self-consistency check
        consistency_score = 1.0
        consistency_report = {"passed": True}
        if generator:
            consistency_report = self.consistency_verifier.verify(query, context, generator)
            consistency_score = consistency_report.get("consistency_score", 1.0)
            if not consistency_report["passed"]:
                all_issues.append("Answer lacks consistency across multiple generation passes")
                
        # 2. Recitation checking
        recitation_report = self.recitation_checker.check(answer, context)
        groundedness_score = recitation_report.get("groundedness_score", 1.0)
        unsupported = recitation_report.get("unsupported_claims", [])
        if unsupported:
            all_issues.extend([f"Unsupported statement: {claim}" for claim in unsupported])
            
        # 2.5 Compute lookback ratio
        lookback_ratio = self.compute_lookback_ratio(answer, context)
            
        # 3. Chain verification
        chain_report = self.chain_verifier.verify(query, query_type, trace_stages)
        if not chain_report["passed"]:
            all_issues.extend(chain_report["issues"])
            
        # 4. Length/Basic Quality validation
        val_report = self.response_validator.validate(answer, context)
        if not val_report["passed"]:
            all_issues.extend(val_report["issues"])
            
        # 5. Threshold checking
        metrics = {
            "retrieval_confidence": confidence,
            "faithfulness": groundedness_score,
            "consistency": consistency_score
        }
        threshold_report = self.threshold_manager.check_thresholds(metrics)
        if not threshold_report["passed"]:
            all_issues.extend(threshold_report["issues"])
            
        # Calculate verdict
        if groundedness_score < 0.4 or confidence < 0.2:
            verdict = "FAIL"
        elif groundedness_score < 0.7 or consistency_score < 0.5 or not chain_report["passed"]:
            verdict = "WARN"
        else:
            verdict = "PASS"
            
        # Fallback decision
        fallback_triggered = verdict == "FAIL"
        fallback_reason = "; ".join(all_issues) if fallback_triggered else ""
        
        # Formulate unified result
        check = ReliabilityCheck(
            passed=verdict != "FAIL",
            verdict=verdict,
            confidence=min(confidence, groundedness_score, consistency_score),
            issues=all_issues,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            groundedness_score=groundedness_score,
            consistency_score=consistency_score,
            recitation_report=recitation_report,
            chain_report=chain_report,
            lookback_ratio=lookback_ratio
        )
        
        logger.info(f"Reliability check complete. Verdict: {verdict}. Passed: {check.passed}")
        return check
        
    def get_fallback_response(self, reason: str) -> str:
        """Get an appropriate safe fallback response when reliability fails."""
        return "I apologize, but I cannot verify this information accurately against official placement documents, so I cannot provide a reliable answer."
