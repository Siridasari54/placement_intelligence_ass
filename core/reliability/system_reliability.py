"""AI System Reliability Layer for self-consistency, validation, and unsupported answer rejection."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityCheck:
    """Result of reliability check."""
    passed: bool
    confidence: float
    issues: List[str]
    fallback_triggered: bool
    fallback_reason: str


class SelfConsistencyVerifier:
    """Verifies self-consistency of generated answers."""
    
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
            sample = generator.generate(query, context)
            samples.append(sample)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency(samples)
        
        # Select most consistent answer
        best_answer = self._select_best_answer(samples)
        
        logger.info(f"Self-consistency score: {consistency_score:.2f}")
        
        return {
            "consistency_score": consistency_score,
            "samples": samples,
            "best_answer": best_answer,
            "passed": consistency_score > 0.6
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
        
        # Simple similarity based on keyword overlap
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
        # Select the answer with highest average similarity to others
        best_idx = 0
        best_score = 0.0
        
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


class ResponseValidator:
    """Validates responses against quality criteria."""
    
    def __init__(self):
        """Initialize the response validator."""
        self.min_length = 50
        self.max_length = 2000
        self.required_elements = ["answer"]
        
        logger.info("ResponseValidator initialized")
    
    def validate(self, response: str, context: List[Document]) -> Dict[str, Any]:
        """Validate response against quality criteria.
        
        Args:
            response: Generated response
            context: Retrieved context
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating response")
        
        issues = []
        
        # Check length
        if len(response) < self.min_length:
            issues.append(f"Response too short: {len(response)} chars (min: {self.min_length})")
        
        if len(response) > self.max_length:
            issues.append(f"Response too long: {len(response)} chars (max: {self.max_length})")
        
        # Check for required elements
        for element in self.required_elements:
            if element not in response.lower():
                issues.append(f"Missing required element: {element}")
        
        # Check for empty or generic responses
        if self._is_generic_response(response):
            issues.append("Response appears generic or placeholder")
        
        # Check grounding in context
        if not self._is_grounded(response, context):
            issues.append("Response not well-grounded in context")
        
        passed = len(issues) == 0
        
        logger.info(f"Validation {'passed' if passed else 'failed'}: {len(issues)} issues")
        
        return {
            "passed": passed,
            "issues": issues,
            "length": len(response)
        }
    
    def _is_generic_response(self, response: str) -> bool:
        """Check if response is generic or placeholder.
        
        Args:
            response: Response text
            
        Returns:
            True if generic, False otherwise
        """
        generic_patterns = [
            "i don't have information",
            "i cannot answer",
            "placeholder",
            "not implemented"
        ]
        
        response_lower = response.lower()
        return any(pattern in response_lower for pattern in generic_patterns)
    
    def _is_grounded(self, response: str, context: List[Document]) -> bool:
        """Check if response is grounded in context.
        
        Args:
            response: Response text
            context: Context documents
            
        Returns:
            True if grounded, False otherwise
        """
        if not context:
            return False
        
        response_words = set(response.lower().split())
        
        for doc in context:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(response_words & doc_words)
            
            if overlap > len(response_words) * 0.3:
                return True
        
        return False


class ConfidenceThresholdManager:
    """Manages confidence thresholds for system reliability."""
    
    def __init__(self):
        """Initialize the confidence threshold manager."""
        self.thresholds = {
            "retrieval_confidence": 0.5,
            "answer_confidence": 0.6,
            "faithfulness": 0.7,
            "consistency": 0.6
        }
        
        logger.info("ConfidenceThresholdManager initialized")
    
    def check_thresholds(self, metrics: Dict[str, float]) -> ReliabilityCheck:
        """Check if metrics meet required thresholds.
        
        Args:
            metrics: Dictionary of metric scores
            
        Returns:
            ReliabilityCheck result
        """
        logger.info("Checking confidence thresholds")
        
        issues = []
        passed = True
        
        for metric_name, threshold in self.thresholds.items():
            metric_value = metrics.get(metric_name, 0.0)
            
            if metric_value < threshold:
                issues.append(f"{metric_name} below threshold: {metric_value:.2f} < {threshold}")
                passed = False
        
        # Determine if fallback should be triggered
        fallback_triggered = not passed
        fallback_reason = " ".join(issues) if issues else ""
        
        # Calculate overall confidence
        overall_confidence = sum(metrics.values()) / len(metrics) if metrics else 0.0
        
        check = ReliabilityCheck(
            passed=passed,
            confidence=overall_confidence,
            issues=issues,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason
        )
        
        logger.info(f"Threshold check {'passed' if passed else 'failed'}")
        return check
    
    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """Set a specific threshold.
        
        Args:
            metric_name: Name of metric
            threshold: Threshold value
        """
        self.thresholds[metric_name] = threshold
        logger.info(f"Set threshold {metric_name} to {threshold}")
    
    def get_threshold(self, metric_name: str) -> float:
        """Get a specific threshold.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Threshold value
        """
        return self.thresholds.get(metric_name, 0.5)


class UnsupportedAnswerRejector:
    """Rejects unsupported answers based on confidence and grounding."""
    
    def __init__(self, confidence_threshold: float = 0.4):
        """Initialize the unsupported answer rejector.
        
        Args:
            confidence_threshold: Minimum confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        logger.info(f"UnsupportedAnswerRejector initialized with threshold {confidence_threshold}")
    
    def should_reject(
        self,
        answer: str,
        confidence: float,
        context: List[Document]
    ) -> Tuple[bool, str]:
        """Determine if answer should be rejected.
        
        Args:
            answer: Generated answer
            confidence: Confidence score
            context: Retrieved context
            
        Returns:
            Tuple of (should_reject, reason)
        """
        logger.info("Checking if answer should be rejected")
        
        reasons = []
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            reasons.append(f"Confidence too low: {confidence:.2f} < {self.confidence_threshold}")
        
        # Check if context is empty
        if not context:
            reasons.append("No context available")
        
        # Check if answer is generic
        if self._is_generic_answer(answer):
            reasons.append("Answer appears generic")
        
        # Check if answer is ungrounded
        if not self._is_grounded(answer, context):
            reasons.append("Answer not grounded in context")
        
        should_reject = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else ""
        
        logger.info(f"Reject decision: {should_reject} (reason: {reason})")
        
        return should_reject, reason
    
    def _is_generic_answer(self, answer: str) -> bool:
        """Check if answer is generic.
        
        Args:
            answer: Answer text
            
        Returns:
            True if generic, False otherwise
        """
        generic_indicators = [
            "i don't have enough information",
            "i cannot answer this question",
            "this is beyond my knowledge",
            "i'm not sure about this"
        ]
        
        answer_lower = answer.lower()
        return any(indicator in answer_lower for indicator in generic_indicators)
    
    def _is_grounded(self, answer: str, context: List[Document]) -> bool:
        """Check if answer is grounded in context.
        
        Args:
            answer: Answer text
            context: Context documents
            
        Returns:
            True if grounded, False otherwise
        """
        if not context:
            return False
        
        answer_words = set(answer.lower().split())
        
        for doc in context[:3]:  # Check top 3 documents
            doc_words = set(doc.page_content.lower().split())
            overlap = len(answer_words & doc_words)
            
            if overlap > len(answer_words) * 0.2:
                return True
        
        return False


class SystemReliabilityLayer:
    """Unified system reliability layer."""
    
    def __init__(self):
        """Initialize the system reliability layer."""
        self.consistency_verifier = SelfConsistencyVerifier()
        self.response_validator = ResponseValidator()
        self.threshold_manager = ConfidenceThresholdManager()
        self.answer_rejector = UnsupportedAnswerRejector()
        
        logger.info("SystemReliabilityLayer initialized")
    
    def check_reliability(
        self,
        query: str,
        answer: str,
        context: List[Document],
        confidence: float,
        generator = None
    ) -> ReliabilityCheck:
        """Perform comprehensive reliability check.
        
        Args:
            query: User query
            answer: Generated answer
            context: Retrieved context
            confidence: Confidence score
            generator: Optional generator for consistency check
            
        Returns:
            ReliabilityCheck result
        """
        logger.info("Performing comprehensive reliability check")
        
        all_issues = []
        
        # Self-consistency check (if generator provided)
        consistency_score = 1.0
        if generator:
            consistency_result = self.consistency_verifier.verify(query, context, generator)
            consistency_score = consistency_result["consistency_score"]
            if not consistency_result["passed"]:
                all_issues.extend([f"Consistency issue: {issue}" for issue in consistency_result.get("issues", [])])
        
        # Response validation
        validation_result = self.response_validator.validate(answer, context)
        if not validation_result["passed"]:
            all_issues.extend(validation_result["issues"])
        
        # Confidence threshold check
        metrics = {
            "retrieval_confidence": confidence,
            "answer_confidence": confidence,
            "faithfulness": 0.8,  # Would come from evaluation engine
            "consistency": consistency_score
        }
        
        threshold_check = self.threshold_manager.check_thresholds(metrics)
        all_issues.extend(threshold_check.issues)
        
        # Unsupported answer check
        should_reject, reject_reason = self.answer_rejector.should_reject(answer, confidence, context)
        if should_reject:
            all_issues.append(f"Unsupported answer: {reject_reason}")
        
        # Determine overall pass/fail
        passed = len(all_issues) == 0
        
        # Calculate overall confidence
        overall_confidence = threshold_check.confidence
        
        check = ReliabilityCheck(
            passed=passed,
            confidence=overall_confidence,
            issues=all_issues,
            fallback_triggered=threshold_check.fallback_triggered or should_reject,
            fallback_reason=threshold_check.fallback_reason or reject_reason
        )
        
        logger.info(f"Reliability check {'passed' if passed else 'failed'}")
        return check
    
    def get_fallback_response(self, reason: str) -> str:
        """Get appropriate fallback response.
        
        Args:
            reason: Reason for fallback
            
        Returns:
            Fallback response
        """
        fallback_responses = {
            "low_confidence": "I'm not confident enough to provide an accurate answer based on the available information.",
            "no_context": "I don't have enough relevant information in the placement documents to answer this question.",
            "generic_answer": "I apologize, but I cannot provide a specific answer for this question.",
            "ungrounded": "I cannot verify this information against the placement documents, so I cannot provide a reliable answer."
        }
        
        # Determine fallback type based on reason
        if "confidence" in reason.lower():
            return fallback_responses["low_confidence"]
        elif "context" in reason.lower():
            return fallback_responses["no_context"]
        elif "generic" in reason.lower():
            return fallback_responses["generic_answer"]
        elif "grounded" in reason.lower():
            return fallback_responses["ungrounded"]
        else:
            return "I apologize, but I cannot provide a reliable answer for this question based on the available information."
