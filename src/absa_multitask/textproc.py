from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


_RE_HTML = re.compile(r"<br\s*/?>", re.IGNORECASE)
_RE_NON_WORD = re.compile(r"[^\w\s'-]+", re.UNICODE)
_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_NUM = re.compile(r"\d+")


def clean_text(text: str, *, remove_numbers: bool = True) -> str:
    t = text or ""
    t = _RE_HTML.sub(" ", t)
    t = t.lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    if remove_numbers:
        t = _RE_NUM.sub(" ", t)
    t = _RE_NON_WORD.sub(" ", t)
    t = _RE_MULTI_SPACE.sub(" ", t).strip()
    return t


def simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.split()


def _try_nltk_pos(tokens: Sequence[str]) -> List[str] | None:
    try:
        import nltk  # noqa: F401
        from nltk import pos_tag
    except Exception:
        return None

    try:
        # Will raise LookupError if tagger not present.
        tagged = pos_tag(list(tokens))
        return [t for _, t in tagged]
    except LookupError:
        return None
    except Exception:
        return None


def pos_tags(tokens: Sequence[str]) -> List[str]:
    """
    Paper dùng Stanford POS Tagger. Ở đây dùng NLTK nếu có;
    nếu môi trường thiếu resource (offline), fallback về tag 'X'.
    """
    if not tokens:
        return []
    tags = _try_nltk_pos(tokens)
    if tags is not None:
        return tags
    return ["X"] * len(tokens)


@dataclass(frozen=True)
class Example:
    tokens: List[str]
    pos: List[str]
    aspect: str
    sentiment: str


def build_examples(
    texts: Iterable[str],
    aspects: Iterable[str],
    sentiments: Iterable[str],
    *,
    remove_numbers: bool = True,
) -> List[Example]:
    out: List[Example] = []
    for text, asp, sent in zip(texts, aspects, sentiments, strict=False):
        cleaned = clean_text(text, remove_numbers=remove_numbers)
        toks = simple_tokenize(cleaned)
        if not toks:
            continue
        pos = pos_tags(toks)
        out.append(Example(tokens=toks, pos=pos, aspect=str(asp), sentiment=str(sent)))
    return out

