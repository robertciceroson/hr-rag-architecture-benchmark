"""
Run with:  pytest -q

1. The control condition must reproduce the published baseline exactly.
2. Every experiment config must run end to end. Dense/rerank/LLM pieces are
   replaced with small offline stubs so CI needs no model downloads or keys.
3. Cross-validation must never tune on the item it scores.
"""
import os
import sys

import numpy as np
import pytest
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rag_bench import calibration  # noqa: E402
from rag_bench.chunking import build_chunks  # noqa: E402
from rag_bench.runner import DATA, load_dataset, run_experiment  # noqa: E402

EXPS = {e["id"]: e for e in yaml.safe_load(
    open(os.path.join(ROOT, "configs", "experiments.yaml")))["experiments"]}


class StubEmbedder:
    """LSA vectors fitted on the policy corpus: a stand-in for bge-small."""
    def __init__(self):
        text = [c.index_text for c in build_chunks(os.path.join(DATA, "policy_docs"), "section")]
        self.vec = TfidfVectorizer(stop_words="english").fit(text)
        self.svd = TruncatedSVD(32, random_state=0).fit(self.vec.transform(text))

    def _emb(self, texts):
        return list(self.svd.transform(self.vec.transform(texts)))

    passage_embed = query_embed = _emb


class StubEncoder:
    def rerank(self, query, docs):
        q = set(query.lower().split())
        return [len(q & set(d.lower().split())) / (1 + len(q)) for d in docs]


class StubLLM:
    calls = 0

    def complete(self, prompt, system=None, max_tokens=300):
        self.calls += 1
        if system:  # scope gate
            return "OUT_OF_SCOPE" if "dress code" in prompt.lower() else "IN_SCOPE"
        return "Employees accrue paid time off each month per company policy."


def test_baseline_reproduces_published_score():
    r = run_experiment(EXPS["E0"], write=False, verbose=False)
    fx = r["fixed"]
    assert (fx["passed"], fx["total"]) == (64, 73)
    assert fx["by_category"]["in_scope"]["passed"] == 43
    assert fx["by_category"]["out_of_scope"]["passed"] == 3


@pytest.mark.parametrize("exp_id", sorted(EXPS, key=lambda k: int(k[1:])))
def test_every_experiment_runs_offline(exp_id):
    exp = EXPS[exp_id]
    if exp_id == "E1" or "bm25" in str(exp["retriever"]):
        pytest.importorskip("rank_bm25")
    r = run_experiment(exp, llm=StubLLM(), embedder=StubEmbedder(),
                       encoder=StubEncoder(), write=False, verbose=False)
    assert r["cv"]["total"] == 73
    # Guardrails are a controlled variable: adversarial must never regress.
    assert r["cv"]["by_category"]["adversarial"]["passed"] == 10


def test_cv_never_sees_held_out_item(monkeypatch):
    feats_seen = []
    real = calibration.search_params

    def spy(feats):
        feats_seen.append({f.item_id for f in feats})
        return real(feats)

    monkeypatch.setattr(calibration, "search_params", spy)
    r = run_experiment(EXPS["E0"], write=False, verbose=False)
    folds = feats_seen[:5]  # first 5 calls are the CV folds
    all_ids = {d["id"] for d in load_dataset()}
    held_out = [all_ids - f for f in folds]
    assert all(len(h) > 0 for h in held_out)
    assert set().union(*held_out) == all_ids          # every item held out once
    assert sum(len(h) for h in held_out) == len(all_ids)
    assert r["cv"]["total"] == 73


# ---------------------------------------------------------------- LLM client ---

class _FakeResp:
    def __init__(self, text, finish="stop"):
        msg = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": msg, "finish_reason": finish})()]


class _FakeClient:
    def __init__(self, text):
        self.text, self.kwargs = text, None
        self.chat = type("Chat", (), {})()
        self.chat.completions = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResp(self.text, "length" if not self.text else "stop")


def test_reasoning_model_gets_token_room(tmp_path):
    from rag_bench.llm import CachedLLM
    client = _FakeClient("IN_SCOPE")
    llm = CachedLLM(str(tmp_path / "c.json"), model="openai/gpt-oss-120b", client=client)
    assert llm.complete("q", max_tokens=5) == "IN_SCOPE"
    assert client.kwargs["max_tokens"] >= 1024
    assert client.kwargs["extra_body"] == {"reasoning_effort": "low"}


def test_empty_answer_raises_and_is_not_cached(tmp_path):
    from rag_bench.llm import CachedLLM
    llm = CachedLLM(str(tmp_path / "c.json"), model="openai/gpt-oss-120b",
                    client=_FakeClient(""))
    with pytest.raises(RuntimeError, match="empty answer"):
        llm.complete("q", max_tokens=5)
    assert not (tmp_path / "c.json").exists()


def test_scope_verdict_parsing_is_strict():
    from rag_bench.gates import llm_scope_check

    class L:
        def __init__(self, out): self.out = out
        def complete(self, *a, **k): return self.out

    assert llm_scope_check(L("OUT_OF_SCOPE"), "q", []) is True
    assert llm_scope_check(L("in scope"), "q", []) is False
    with pytest.raises(ValueError):
        llm_scope_check(L("Sure! Happy to help."), "q", [])
