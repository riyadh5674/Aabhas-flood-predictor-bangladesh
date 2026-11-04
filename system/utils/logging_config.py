"""
Structured logging configuration for Aabhas Flood Predictor.

Provides JSON-formatted logs with consistent fields for Cloud Logging,
Elasticsearch, or other log aggregation systems.
"""
from __future__ import annotations

import json
import logging
import logging.config
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
JSON_LOG_FORMAT = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

# Request ID context variable
_request_id: Optional[str] = None


def set_request_id(request_id: str) -> None:
    """Set the current request ID for logging context."""
    global _request_id
    _request_id = request_id


def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return _request_id


class RequestIdFilter(logging.Filter):
    """Add request ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request ID if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "message", "pathname", "process", "processName",
                    "relativeCreated", "thread", "threadName", "exc_info",
                    "exc_text", "stack_info", "request_id"
                }:
                    try:
                        json.dumps(value)  # Test serializability
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        return json.dumps(log_data, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON formatting (for Cloud Logging, ELK, etc.)
        log_file: Optional file path for file logging
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set level
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestIdFilter())
        root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    
    # Log startup
    logging.getLogger("aabhas.startup").info(
        f"Logging configured: level={level}, json={json_format}, file={log_file}"
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the aabhas namespace."""
    return logging.getLogger(f"aabhas.{name}")


# Context manager for request logging
class RequestLoggingContext:
    """Context manager for request-scoped logging."""
    
    def __init__(self, request_id: str, logger: Optional[logging.Logger] = None):
        self.request_id = request_id
        self.logger = logger or get_logger("request")
        self._previous_id = get_request_id()
    
    def __enter__(self):
        set_request_id(self.request_id)
        self.logger.info("Request started", extra={"request_id": self.request_id})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(
                "Request failed",
                extra={
                    "request_id": self.request_id,
                    "exception": str(exc_val),
                    "exception_type": exc_type.__name__,
                }
            )
        else:
            self.logger.info("Request completed", extra={"request_id": self.request_id})
        set_request_id(self._previous_id)


# Initialize default logging
if not logging.getLogger().handlers:
    configure_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        json_format=os.getenv("LOG_JSON", "").lower() == "true",
        log_file=os.getenv("LOG_FILE"),
    )