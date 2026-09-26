"""
Rule-based guardrails, carried over UNCHANGED from hr-policy-eval-harness
(src/pipeline.py). They are a controlled variable in this benchmark: every
experiment uses the same injection and ambiguity-pattern rules, so any change
in score comes from retrieval and scope gating, not from edited guardrails.

Because the injection check runs before retrieval, the 10 adversarial items
act as a regression check: they should score 10/10 in every experiment.
"""
import re

INJECTION_PATTERNS = [
    r"ignore (your |the )?(previous|prior|above) instructions",
    r"disregard (the |your )?.*(section|policy|rules?|instructions)",
    r"\bpretend\b",
    r"you('re| are) now in (an? )?(unrestricted|developer|admin) mode",
    r"system prompt override",
    r"reveal your (internal )?(instructions|system prompt)",
    r"as (my|your) (administrator|admin)",
    r"i (have|'ve got) (hr )?admin access",
    r"just make up",
    r"make up a (reasonable|plausible)",
    r"write me a fake",
    r"auto-?approve",
]

PERSONAL_DATA_PATTERNS = [
    r"\b(salary|compensation|pto balance|pay)\b.*\b(of|for)\b\s+[A-Z][a-z]+\s+[A-Z][a-z]+",
    r"\bCEO'?s?\s+(exact\s+)?salary\b",
    r"salary bands? for (every|each|all)",
]

NAMED_PERSON_PATTERN = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+('s)?\b")
NAMED_PERSON_KEYWORDS = re.compile(
    r"\b(pto balance|salary|compensation|disciplinary record|pay)\b", re.IGNORECASE)

AMBIGUOUS_PATTERNS = [
    r"^how much (pto|leave|time off) do i have\b",
    r"^can i take leave\b",
    r"^what'?s the reimbursement limit\??$",
    r"^am i eligible\??$",
    r"^how do i enroll\??$",
    r"^what'?s the approval process\??$",
    r"^when does my coverage start\??$",
    r"^can i get reimbursed for this\??$",
]

# The original harness's hand-tuned values (tuned on TF-IDF score scale).
ORIGINAL_PARAMS = {"oos_threshold": 0.10, "amb_margin": 0.02, "amb_min_conf": 0.20}


def is_injection(query: str) -> bool:
    q = query.lower()
    if any(re.search(p, q) for p in INJECTION_PATTERNS):
        return True
    if any(re.search(p, query) for p in PERSONAL_DATA_PATTERNS):
        return True
    return bool(NAMED_PERSON_PATTERN.search(query) and NAMED_PERSON_KEYWORDS.search(query))


def is_pattern_ambiguous(query: str) -> bool:
    q = query.lower().strip()
    return any(re.search(p, q) for p in AMBIGUOUS_PATTERNS)
