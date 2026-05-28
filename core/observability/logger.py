"""Unified logging infrastructure with structured logging and observability."""

import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self):
        """Initialize the structured formatter."""
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


class ObservabilityLogger:
    """Unified logger with observability features."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize the observability logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
        # File handler for structured logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message.
        
        Args:
            message: Log message
            **kwargs: Additional context
        """
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message.
        
        Args:
            message: Log message
            **kwargs: Additional context
        """
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message.
        
        Args:
            message: Log message
            exc_info: Include exception info
            **kwargs: Additional context
        """
        self.logger.error(message, exc_info=exc_info, extra=kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message.
        
        Args:
            message: Log message
            **kwargs: Additional context
        """
        self.logger.debug(message, extra=kwargs)


class MetricsCollector:
    """Collects and tracks system metrics."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics: Dict[str, Any] = {
            "query_count": 0,
            "total_latency_ms": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }
    
    def increment_query_count(self) -> None:
        """Increment query count."""
        self.metrics["query_count"] += 1
    
    def record_latency(self, latency_ms: float) -> None:
        """Record query latency.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self.metrics["total_latency_ms"] += latency_ms
    
    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.metrics["cache_hits"] += 1
    
    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.metrics["cache_misses"] += 1
    
    def record_error(self) -> None:
        """Record error."""
        self.metrics["errors"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        avg_latency = (
            self.metrics["total_latency_ms"] / self.metrics["query_count"]
            if self.metrics["query_count"] > 0
            else 0
        )
        
        cache_hit_rate = (
            self.metrics["cache_hits"] / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
            else 0
        )
        
        return {
            **self.metrics,
            "avg_latency_ms": avg_latency,
            "cache_hit_rate": cache_hit_rate
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            "query_count": 0,
            "total_latency_ms": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()
