from .agency import DelegationScope, ExcessiveAgency, OversightBypass, UndeclaredTool
from .grounded import Grounded
from .harmful_output import HarmfulOutput
from .honesty import ActionClaims, ToolErrorMasking
from .injection import IndirectInjection
from .phi import PHI
from .pii import PII
from .reliability import GoalCompletion, LoopDetection, RunStats
from .security import DataExfiltration, GoalHijacking, PromptInjection, SecretsExposure, SystemPromptLeakage
from .tool_risk import ToolRisk

ALL_RULES = (
    # security
    PromptInjection, IndirectInjection, GoalHijacking, SystemPromptLeakage, SecretsExposure, HarmfulOutput,
    # privacy
    PII, PHI, DataExfiltration,
    # action safety
    ToolRisk, ExcessiveAgency, DelegationScope, OversightBypass, UndeclaredTool,
    # honesty
    Grounded, ActionClaims, ToolErrorMasking,
    # reliability
    GoalCompletion, LoopDetection, RunStats,
)

__all__ = [c.__name__ for c in ALL_RULES] + ["ALL_RULES"]
