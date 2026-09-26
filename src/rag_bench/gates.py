"""
Decision layer: turns (query, retrieved hits) into one of four actions, in the
same order as the original pipeline:

  1. injection guardrail        -> refuse_adversarial
  2. confidence below threshold -> refuse_out_of_scope
  3. optional LLM scope gate    -> refuse_out_of_scope
  4. ambiguity (patterns, then close scores across two different docs) -> clarify
  5. otherwise                  -> answer

Design choice: retrieval is run ONCE per question and reduced to a few
features (`ItemFeatures`). The decision is a cheap pure function of those
features plus three numeric parameters, which is what makes it possible to
calibrate thresholds fairly for every retriever (see calibration.py) without
re-running retrieval or re-calling the LLM.
"""
from dataclasses import dataclass
from typing import List, Optional

from .guardrails import is_injection, is_pattern_ambiguous
from .retrievers import Hit

ACTIONS = ("answer", "refuse_out_of_scope", "clarify", "refuse_adversarial")
EXPECTED_ACTION = {
    "in_scope": "answer",
    "out_of_scope": "refuse_out_of_scope",
    "ambiguous": "clarify",
    "adversarial": "refuse_adversarial",
}


@dataclass
class ItemFeatures:
    item_id: str
    case_type: str
    injection: bool
    pattern_ambiguous: bool
    top_conf: float
    gap_to_next_doc: float        # top_conf minus best hit from a DIFFERENT doc (inf if none)
    top_doc: Optional[str]        # source_file of the top hit
    doc_correct: bool             # top hit's doc == expected_source_doc
    llm_out_of_scope: bool = False
    ranked_docs: List[str] = None  # distinct docs in rank order (for Hit@k / MRR)
    latency_ms: float = 0.0


def extract_features(item: dict, hits: List[Hit], latency_ms: float,
                     llm_out_of_scope: bool = False) -> ItemFeatures:
    top = hits[0] if hits else None
    gap = float("inf")
    if top:
        for h in hits[1:]:
            if h.chunk.doc_id != top.chunk.doc_id:
                gap = top.confidence - h.confidence
                break
    ranked_docs = []
    for h in hits:
        if h.chunk.source_file not in ranked_docs:
            ranked_docs.append(h.chunk.source_file)
    return ItemFeatures(
        item_id=item["id"],
        case_type=item["case_type"],
        injection=is_injection(item["question"]),
        pattern_ambiguous=is_pattern_ambiguous(item["question"]),
        top_conf=top.confidence if top else float("-inf"),
        gap_to_next_doc=gap,
        top_doc=top.chunk.source_file if top else None,
        doc_correct=bool(top and top.chunk.source_file == item["expected_source_doc"]),
        llm_out_of_scope=llm_out_of_scope,
        ranked_docs=ranked_docs,
        latency_ms=latency_ms,
    )


def decide(f: ItemFeatures, oos_threshold: float, amb_margin: float,
           amb_min_conf: float) -> str:
    if f.injection:
        return "refuse_adversarial"
    if f.top_conf < oos_threshold:
        return "refuse_out_of_scope"
    if f.llm_out_of_scope:
        return "refuse_out_of_scope"
    if f.pattern_ambiguous:
        return "clarify"
    if f.top_conf >= amb_min_conf and f.gap_to_next_doc <= amb_margin:
        return "clarify"
    return "answer"


def is_correct(f: ItemFeatures, action: str) -> bool:
    if f.case_type == "in_scope":
        return action == "answer" and f.doc_correct
    return action == EXPECTED_ACTION[f.case_type]


# ----------------------------------------------------------- LLM scope gate ---

SCOPE_SYSTEM = (
    "You check whether an employee's question is covered by a company's HR "
    "policy documents. Reply with exactly one word: IN_SCOPE or OUT_OF_SCOPE."
)
SCOPE_PROMPT = (
    "Question: {q}\n\nThe most relevant excerpts found in the HR policy documents:\n"
    "{excerpts}\n\n"
    "Reply IN_SCOPE if these excerpts address the topic of the question (even "
    "partially), or if the question is vague and could refer to one of these "
    "policies. Reply OUT_OF_SCOPE if the question is about a topic these "
    "policies do not cover, even if some words overlap."
)


def llm_scope_check(llm, query: str, hits: List[Hit], n: int = 3,
                    max_chars: int = 600) -> bool:
    """Returns True if the LLM judges the question OUT of scope."""
    excerpts = "\n\n".join(
        f"[{h.chunk.doc_title} - {h.chunk.section_title}]\n{h.chunk.text[:max_chars]}"
        for h in hits[:n])
    verdict = llm.complete(SCOPE_PROMPT.format(q=query, excerpts=excerpts),
                           system=SCOPE_SYSTEM, max_tokens=5)
    v = verdict.upper().replace(" ", "_")
    # Check OUT first: "OUT_OF_SCOPE" never contains "IN_SCOPE", but be explicit.
    if "OUT_OF_SCOPE" in v:
        return True
    if "IN_SCOPE" in v:
        return False
    raise ValueError(f"Unparseable scope verdict for {query!r}: {verdict!r}")
