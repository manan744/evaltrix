"""evaltrix: evals and governance for agent traces, judged by Jev."""

from .engine import TraceReport, evaluate, evaluate_many, evaluate_sync
from .evals import Eval, Result
from .jev import Choice, FakeJev, Jev, JevError, Score, Usage, YesNo
from .trace import Trace, Unit

__version__ = "0.0.1"

__all__ = [
    "Choice", "Eval", "FakeJev", "Jev", "JevError", "Result", "Score", "Trace", "TraceReport", "Unit",
    "Usage", "YesNo", "evaluate", "evaluate_many", "evaluate_sync",
]
