"""
Chunking strategies (AI Engineering Guidebook, "5 chunking strategies for RAG", p.118).

Every strategy returns a list of `Chunk` objects that carry their source
document, so retrieval can always be scored at the document level no matter
how the text was split.

  section          - split on "## " headers. Identical to the original
                     hr-policy-eval-harness pipeline (the control condition).
  section_contextual - section chunks with the document title prepended
                     ("contextual chunk headers"). Tests whether telling the
                     retriever *which policy* a section belongs to fixes
                     cross-document confusion (e.g. "leave the company" vs.
                     "parental leave").
  fixed            - fixed-size word windows with overlap, ignoring document
                     structure. The naive default in many tutorials.
  sentence_window  - small windows of consecutive sentences within a section,
                     prefixed with the section title. Finer-grained matching.
"""
import glob
import os
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    doc_title: str
    section_title: str
    text: str

    @property
    def index_text(self) -> str:
        """The text the retriever actually indexes."""
        return f"{self.section_title}. {self.text}" if self.section_title else self.text


def _read_docs(policy_docs_dir: str):
    for path in sorted(glob.glob(os.path.join(policy_docs_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        filename = os.path.basename(path)
        doc_id_match = re.search(r"\*\*Document ID:\*\*\s*(\S+)", content)
        doc_id = doc_id_match.group(1) if doc_id_match else filename
        title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else filename
        yield filename, doc_id, doc_title, content


def _sections(content: str):
    for section in re.split(r"\n(?=## )", content):
        section = section.strip()
        if not section.startswith("## "):
            continue
        title_line, *rest = section.split("\n", 1)
        body = rest[0].strip() if rest else ""
        if body:
            yield title_line.replace("## ", "").strip(), body


def chunk_section(policy_docs_dir: str, contextual: bool = False) -> List[Chunk]:
    chunks = []
    for filename, doc_id, doc_title, content in _read_docs(policy_docs_dir):
        for i, (title, body) in enumerate(_sections(content)):
            section_title = f"{doc_title} - {title}" if contextual else title
            chunks.append(Chunk(f"{filename}#s{i}", doc_id, filename, doc_title,
                                section_title, body))
    return chunks


def chunk_fixed(policy_docs_dir: str, size: int = 120, overlap: int = 30) -> List[Chunk]:
    chunks = []
    step = max(1, size - overlap)
    for filename, doc_id, doc_title, content in _read_docs(policy_docs_dir):
        words = content.split()
        for i, start in enumerate(range(0, max(1, len(words) - overlap), step)):
            text = " ".join(words[start:start + size])
            if text:
                chunks.append(Chunk(f"{filename}#f{i}", doc_id, filename, doc_title, "", text))
    return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+(?=[-*\d])")


def chunk_sentence_window(policy_docs_dir: str, window: int = 3, stride: int = 2) -> List[Chunk]:
    chunks = []
    for filename, doc_id, doc_title, content in _read_docs(policy_docs_dir):
        for s_idx, (title, body) in enumerate(_sections(content)):
            sents = [s.strip() for s in _SENT_SPLIT.split(body) if s.strip()]
            starts = range(0, max(1, len(sents) - window + stride), stride)
            for w_idx, start in enumerate(starts):
                text = " ".join(sents[start:start + window])
                if text:
                    chunks.append(Chunk(f"{filename}#s{s_idx}w{w_idx}", doc_id, filename,
                                        doc_title, title, text))
    return chunks


def build_chunks(policy_docs_dir: str, strategy: str, **params) -> List[Chunk]:
    if strategy == "section":
        return chunk_section(policy_docs_dir, contextual=False)
    if strategy == "section_contextual":
        return chunk_section(policy_docs_dir, contextual=True)
    if strategy == "fixed":
        return chunk_fixed(policy_docs_dir, **params)
    if strategy == "sentence_window":
        return chunk_sentence_window(policy_docs_dir, **params)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
