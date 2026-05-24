"""
Unified pipeline trace — single trace_id per request.

V3 final polish:
  - OTLP-compatible field names (traceId, spanId, parentSpanId, serviceName)
  - Full 32-hex-char trace IDs, 16-hex-char span IDs (OpenTelemetry standard)
  - span.kind = "internal" for pipeline stages
  - status_code per span (OK / ERROR)
  - Error capture per stage (exception type + message)
  - Schema versioned (trace_version=2.0)
  - config_hash embedded for reproducibility
  - Decoupled from pipeline — plug-and-play
"""
import time
import uuid
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass, field
from app.config import CONFIG_HASH, OTEL_SERVICE_NAME, OTEL_SERVICE_VERSION


TRACE_SCHEMA_VERSION = "2.0"


def _hex_id(length: int = 32) -> str:
    """Generate hex ID compatible with OpenTelemetry."""
    return uuid.uuid4().hex[:length]


@dataclass
class StageRecord:
    """A single pipeline stage execution record (OTLP span)."""
    name: str
    span_id: str             # 16-hex-char (OTLP spanId)
    parent_span_id: str      # trace_id or parent span (OTLP parentSpanId)
    start_time: str          # ISO 8601
    end_time: str            # ISO 8601
    latency_ms: float
    status_code: str = "OK"  # "OK" or "ERROR"
    data: dict = field(default_factory=dict)
    error: dict | None = None  # {type, message, traceback} if exception


class PipelineTrace:
    """
    Accumulates a full pipeline trace for one request.
    Produces OTLP-compatible output for OpenTelemetry exporters.

    Usage:
        trace = PipelineTrace()
        with trace.stage("retrieval", {"top_k": 15}) as s:
            results = do_retrieval()
            s["candidate_count"] = len(results)
        output = trace.finalize()
    """

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or _hex_id(32)
        self.trace_version = TRACE_SCHEMA_VERSION
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._stages: list[StageRecord] = []
        self._metadata: dict = {}
        self._start = time.perf_counter()

    def stage(self, name: str, initial_data: dict | None = None):
        """Context manager for timing a pipeline stage."""
        return _StageContext(self, name, initial_data or {})

    def add_metadata(self, key: str, value):
        """Add top-level metadata to the trace."""
        self._metadata[key] = value

    def _record_stage(self, record: StageRecord):
        """Internal: append a completed stage record."""
        self._stages.append(record)

    def get_latency(self, stage_name: str) -> float:
        """Get latency for a specific stage (0.0 if not found)."""
        for s in self._stages:
            if s.name == stage_name:
                return s.latency_ms
        return 0.0

    def get_stage_data(self, stage_name: str) -> dict:
        """Get data dict for a specific stage."""
        for s in self._stages:
            if s.name == stage_name:
                return s.data
        return {}

    def has_errors(self) -> bool:
        """Check if any stage recorded an error."""
        return any(s.error is not None for s in self._stages)

    def finalize(self) -> dict:
        """
        Produce the final structured trace dict.
        OTLP-compatible: traceId, spans with spanId/parentSpanId, resource attributes.
        """
        total_ms = (time.perf_counter() - self._start) * 1000

        latency_ms = {}
        for s in self._stages:
            latency_ms[s.name] = round(s.latency_ms, 2)

        # OTLP-style spans
        spans = []
        for s in self._stages:
            span = {
                "spanId": s.span_id,
                "parentSpanId": s.parent_span_id,
                "name": s.name,
                "kind": "SPAN_KIND_INTERNAL",
                "startTimeUnixNano": s.start_time,
                "endTimeUnixNano": s.end_time,
                "durationMs": round(s.latency_ms, 2),
                "status": {"code": s.status_code},
                "attributes": s.data,
            }
            if s.error:
                span["status"]["message"] = s.error.get("message", "")
                span["events"] = [{
                    "name": "exception",
                    "attributes": {
                        "exception.type": s.error.get("type", ""),
                        "exception.message": s.error.get("message", ""),
                        "exception.stacktrace": s.error.get("traceback", ""),
                    }
                }]
            spans.append(span)

        return {
            # OTLP resource
            "resource": {
                "service.name": OTEL_SERVICE_NAME,
                "service.version": OTEL_SERVICE_VERSION,
                "config_hash": CONFIG_HASH,
            },
            # Trace envelope
            "traceId": self.trace_id,
            "trace_version": self.trace_version,
            "created_at": self.created_at,
            "total_latency_ms": round(total_ms, 2),
            "has_errors": self.has_errors(),
            "latency_ms": latency_ms,
            "spans": spans,
            # Legacy compat
            "trace_id": self.trace_id,
            "stages": spans,
            **self._metadata,
        }


class _StageContext:
    """Context manager that times a stage, captures errors, assigns OTLP spanId."""

    def __init__(self, trace: PipelineTrace, name: str, data: dict):
        self._trace = trace
        self._name = name
        self._data = data
        self._start_time = None
        self._start_perf = None
        self._span_id = _hex_id(16)

    def __enter__(self):
        self._start_time = datetime.now(timezone.utc).isoformat()
        self._start_perf = time.perf_counter()
        return self._data  # caller can mutate this dict

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_perf = time.perf_counter()
        end_time = datetime.now(timezone.utc).isoformat()
        latency = (end_perf - self._start_perf) * 1000

        error = None
        status_code = "OK"
        if exc_type is not None:
            status_code = "ERROR"
            error = {
                "type": exc_type.__name__,
                "message": str(exc_val),
                "traceback": traceback.format_exc()[-500:],
            }

        record = StageRecord(
            name=self._name,
            span_id=self._span_id,
            parent_span_id=self._trace.trace_id,
            start_time=self._start_time,
            end_time=end_time,
            latency_ms=latency,
            status_code=status_code,
            data=self._data,
            error=error,
        )
        self._trace._record_stage(record)
        return False  # don't suppress exceptions
