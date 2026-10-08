"""Deterministic PII candidate detection: regexes plus checksums, no dependencies.

This is the code half of the PII eval. It finds *candidates* cheaply and with high recall; Jev then
decides whether each one is personal data about a private individual or something harmless (a
support address, a documented example, a company phone line). Checksums (Luhn for cards, mod-97 for
IBANs, SSN area rules) remove most random-number false positives before Jev is asked anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    type: str  # EMAIL | PHONE | CREDIT_CARD | US_SSN | IBAN
    value: str
    start: int
    end: int


_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
# +country (optional), area code, number; separators space . - ; needs 10+ digits overall
_PHONE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\w)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_SSN = re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_ok(number: str) -> bool:
    d = [int(c) for c in _digits(number)][::-1]
    total = sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])
    return len(d) >= 13 and total % 10 == 0


def ssn_ok(area: str, group: str, serial: str) -> bool:
    return not (area in ("000", "666") or area.startswith("9") or group == "00" or serial == "0000")


def iban_ok(iban: str) -> bool:
    s = iban.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


def find_pii(text: str) -> list[Entity]:
    """Candidates in text order. A span claimed by a stronger type (card, SSN, IBAN) isn't reported as a phone."""
    found: list[Entity] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    def add(kind: str, m: re.Match[str]) -> None:
        found.append(Entity(kind, m.group(0).strip(), m.start(), m.end()))
        taken.append((m.start(), m.end()))

    for m in _EMAIL.finditer(text):
        add("EMAIL", m)
    # most specific first: an IBAN's digits would otherwise look like a card number
    for m in _IBAN.finditer(text):
        if free(m.start(), m.end()) and iban_ok(m.group(0)):
            add("IBAN", m)
    for m in _CARD.finditer(text):
        if not free(m.start(), m.end()):
            continue
        if luhn_ok(m.group(0)):
            add("CREDIT_CARD", m)
        else:
            # a long digit run that isn't a card (order or tracking number): claim the span anyway so
            # no shorter phone match is carved out of it
            taken.append((m.start(), m.end()))
    for m in _SSN.finditer(text):
        if free(m.start(), m.end()) and ssn_ok(*m.groups()):
            add("US_SSN", m)
    for m in _PHONE.finditer(text):
        if free(m.start(), m.end()) and 10 <= len(_digits(m.group(0))) <= 15:
            add("PHONE", m)
    return sorted(found, key=lambda e: e.start)


def redact(text: str, entities: list[Entity]) -> str:
    for e in sorted(entities, key=lambda e: e.start, reverse=True):
        text = text[: e.start] + f"[{e.type}]" + text[e.end:]
    return text


# --------------------------------------------------------------------------- secrets
#
# Secrets are never sent to the judge. find_secrets() returns candidates; describe_secret() turns one
# into a masked shape (type, length, character classes, entropy) that is enough to tell a live
# credential from "sk-xxxxxxxx" or "<your-api-key>" without revealing it.

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)")),
    ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("OPENAI_KEY", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("STRIPE_KEY", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("BEARER_TOKEN", re.compile(r"(?<=Bearer )[A-Za-z0-9._~+/-]{16,}=*")),
    ("PREFIXED_KEY", re.compile(r"\b[a-z]{2,12}(?:key|token)_[A-Za-z0-9_]{24,}")),
    # key = value assignments: only the value is the candidate
    ("ASSIGNED_SECRET", re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|password|passwd|pwd)"
        r"[\"']?\s*[:=]\s*[\"']?(?P<v>[^\s\"',;}{)]{6,})")),
]

_PLACEHOLDER = re.compile(r"x{4,}|\*{3,}|\.{3}|<[^>]*>|\{\{|your[_-]|example|dummy|placeholder|changeme|redacted|test123",
                          re.I)


def find_secrets(text: str) -> list[Entity]:
    found: list[Entity] = []
    taken: list[tuple[int, int]] = []
    for kind, pat in _SECRET_PATTERNS:
        for m in pat.finditer(text):
            start, end = (m.start("v"), m.end("v")) if "v" in pat.groupindex else (m.start(), m.end())
            if any(not (end <= s or start >= e) for s, e in taken):
                continue
            found.append(Entity(kind, text[start:end], start, end))
            taken.append((start, end))
    return sorted(found, key=lambda e: e.start)


def mask(value: str) -> str:
    """Reveal at most ~1/6 of the value (enough to see a vendor prefix like "sk-p" on long keys)."""
    head = min(4, len(value) // 6)
    tail = 2 if len(value) >= 20 else 0
    return f"{value[:head]}…{value[-tail:] if tail else ''}({len(value)})"


def shannon_entropy(value: str) -> float:
    """Bits per character. Random keys sit around 4.5-6; words and placeholders much lower."""
    import math
    from collections import Counter

    n = len(value)
    return -sum(c / n * math.log2(c / n) for c in Counter(value).values()) if n else 0.0


def describe_secret(e: Entity) -> dict[str, object]:
    v = e.value
    classes = [name for name, pat in (("lower", r"[a-z]"), ("upper", r"[A-Z]"), ("digit", r"\d"), ("symbol", r"[^A-Za-z0-9]"))
               if re.search(pat, v)]
    return {"type": e.type, "masked": mask(v), "length": len(v), "character_classes": classes,
            "entropy_bits_per_char": round(shannon_entropy(v), 2),
            "placeholder_markers": bool(_PLACEHOLDER.search(v))}


def mask_secrets(text: str, entities: list[Entity]) -> str:
    for i, e in sorted(enumerate(entities), key=lambda x: x[1].start, reverse=True):
        text = text[: e.start] + f"[SECRET_{i}:{mask(e.value)}]" + text[e.end:]
    return text


# --------------------------------------------------------------------------- health information
#
# Candidates only: a term list cannot tell "diabetes" in a public FAQ from "Maria's diabetes". The PHI rule
# asks Jev whether health information is tied to an identifiable person; this just decides whether to ask.

_HEALTH_TERMS = re.compile(
    r"\b(diagnos(?:is|ed|es)|prescri(?:ption|bed)|medication|dosage|mg\b|insulin|metformin|chemotherapy|chemo|"
    r"radiation therapy|dialysis|biopsy|mri|ct scan|x-ray|blood (?:test|pressure|sugar)|cholesterol|"
    r"diabetes|diabetic|cancer|tumou?r|hiv|aids|hepatitis|asthma|epilepsy|seizures?|stroke|heart (?:attack|disease|"
    r"failure)|arrhythmia|depression|anxiety|bipolar|schizophrenia|ptsd|adhd|autism|dementia|alzheimer'?s|"
    r"pregnan(?:t|cy)|miscarriage|abortion|ivf|std|sti|herpes|chlamydia|addiction|rehab|overdose|opioid|"
    r"surgery|surgical|post-op|physical therapy|therapy session|psychiatr\w*|oncolog\w*|cardiolog\w*|"
    r"allerg(?:y|ies|ic)|anaphylaxis|disability|wheelchair|hospitali[sz]ed|admitted to|discharged|"
    r"mrn|medical record|lab results?|test results?|positive for|negative for|"
    # clinical context: being someone's patient is itself health information
    r"patients?|follow-up|check-?up|consultation|clinic|doctor|dr\.|nurse|symptoms?|treatment|condition)\b",
    re.I,
)
_MRN = re.compile(r"\b(?:MRN|medical record (?:no\.?|number))[:#\s]*[A-Z0-9-]{5,}\b", re.I)


def find_health_terms(text: str) -> list[Entity]:
    found = [Entity("MEDICAL_RECORD_NUMBER", m.group(0), m.start(), m.end()) for m in _MRN.finditer(text)]
    taken = [(e.start, e.end) for e in found]
    for m in _HEALTH_TERMS.finditer(text):
        if all(m.end() <= s or m.start() >= e for s, e in taken):
            found.append(Entity("HEALTH_TERM", m.group(0), m.start(), m.end()))
    return sorted(found, key=lambda e: e.start)
