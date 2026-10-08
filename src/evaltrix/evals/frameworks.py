"""Standards the built-in rules produce evidence for. Tags are strings so policies can add their own.

OWASP Top 10 for LLM Applications (2025) and OWASP "Agentic AI - Threats and Mitigations" (T1..T15),
plus the EU AI Act article on human oversight. A tag says a rule's findings are *relevant evidence*
for that item, not that running the rule satisfies it.
"""

LLM01_PROMPT_INJECTION = "OWASP-LLM01:2025 Prompt Injection"
LLM02_SENSITIVE_INFO = "OWASP-LLM02:2025 Sensitive Information Disclosure"
LLM06_EXCESSIVE_AGENCY = "OWASP-LLM06:2025 Excessive Agency"
LLM07_SYSTEM_PROMPT_LEAKAGE = "OWASP-LLM07:2025 System Prompt Leakage"
LLM09_MISINFORMATION = "OWASP-LLM09:2025 Misinformation"
LLM10_UNBOUNDED_CONSUMPTION = "OWASP-LLM10:2025 Unbounded Consumption"

T2_TOOL_MISUSE = "OWASP-Agentic:T2 Tool Misuse"
T3_PRIVILEGE_COMPROMISE = "OWASP-Agentic:T3 Privilege Compromise"
T4_RESOURCE_OVERLOAD = "OWASP-Agentic:T4 Resource Overload"
T5_CASCADING_HALLUCINATION = "OWASP-Agentic:T5 Cascading Hallucination"
T6_GOAL_MANIPULATION = "OWASP-Agentic:T6 Intent Breaking & Goal Manipulation"
T7_DECEPTIVE_BEHAVIOR = "OWASP-Agentic:T7 Misaligned & Deceptive Behaviors"
T12_AGENT_COMMUNICATION = "OWASP-Agentic:T12 Agent Communication Poisoning"

EU_AI_ACT_ART14_OVERSIGHT = "EU-AI-Act:Art.14 Human Oversight"

HIPAA_DISCLOSURE = "HIPAA:45 CFR 164.502 Uses and disclosures of PHI"
NIST_600_1_HARMFUL_CONTENT = "NIST-AI-600-1:Dangerous, Violent, or Hateful Content"
NIST_600_1_INFOSEC = "NIST-AI-600-1:Information Security"
