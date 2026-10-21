"""Final-answer extraction and the conservative gold-matching rule ("conservative_norm_v1").

Extraction (``extract_answer``)
  * Only the text inside ``<answer>...</answer>`` is the final answer. Numbers elsewhere in the response are ignored.
  * Status values (anything other than ``ok`` means "no usable final answer"):
      ok, truncated (hit the generation limit without EOS; takes precedence), no_tag, malformed_tag
      (unbalanced tags), multiple_tags, empty, multiple_answers (``or`` / ``;`` / ``\\n`` separated
      alternatives, or two or more numbers inside the tag).

Normalization (``normalize``)
  1. NFKC, lowercase, collapse whitespace, strip surrounding quotes / trailing period, unicode minus -> ``-``.
  2. Numeric form: ``[sign] [currency $€£¥] number [%] [unit words]``.
     * Thousands separators (comma, or a single space as printed on Statista charts) are removed only with strict
       3-digit grouping (``1,234,567`` / ``1 016 529``). ``4,5`` and ``45 6`` are NOT numbers.
     * Values are compared exactly as written with Decimal (``12.0 == 12``, ``12.05 != 12.1``). No rounding.
     * Magnitude words (thousand/million/k/bn, ...) are kept as a unit; they are NEVER multiplied out.
     * ``%`` is never converted (``45% != 0.45``).
     * Currency symbols are dropped (they never change magnitude).
  3. Otherwise the answer is text, compared by exact normalized string equality.

Gold match (``match_gold``): correct iff the answer matches at least one label:
  * numbers: equal Decimal value AND
      - percent: same on both sides, or one side has ``%`` and the question is percentage-related
        (percent/percentage/%/share/rate/proportion);
      - units: equal, or present on exactly one side (the other side has none).
  * text: equal normalized string.
  This is NOT the official ChartQA relaxed-accuracy metric (5% tolerance); results are labelled with ``RULE_NAME``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

RULE_NAME = "conservative_norm_v1"

OK = "ok"
PARSE_STATUSES = ("ok", "truncated", "no_tag", "malformed_tag", "multiple_tags", "empty", "multiple_answers")

_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_OPEN_RE = re.compile(r"<answer>", re.I)
_CLOSE_RE = re.compile(r"</answer>", re.I)
_NUM_TOKEN_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d{1,3}(?: \d{3})+(?![\d])|\d+)(?:\.\d+)?(?![\d])|(?<![\w\d])[-+]?\.\d+")
_ALT_SEP_RE = re.compile(r"\bor\b|;|\n")

_NUMERIC_RE = re.compile(
    r"""^(?P<sign>[-+])?\s*
        (?P<cur>[$€£¥])?\s*
        (?P<num>(?:\d{1,3}(?:,\d{3})+|\d{1,3}(?:\ \d{3})+(?![\d])|\d+)(?:\.\d+)?|\.\d+)\s*
        (?P<pct>%|percent\b|per\s+cent\b)?\s*
        (?P<unit>[a-z][a-z.\s]{0,40})?$""",
    re.X,
)
_PERCENT_QUESTION_RE = re.compile(r"percent|percentage|%|\bshare\b|\brate\b|proportion", re.I)


@dataclass
class Extraction:
    status: str
    answer: str | None          # raw text inside the tag (only meaningful when status == ok)
    n_tags: int = 0


def extract_answer(text: str, truncated: bool = False) -> Extraction:
    tags = _TAG_RE.findall(text)
    n_open, n_close = len(_OPEN_RE.findall(text)), len(_CLOSE_RE.findall(text))
    if truncated:
        return Extraction("truncated", tags[0].strip() if len(tags) == 1 else None, len(tags))
    if n_open == 0 and n_close == 0:
        return Extraction("no_tag", None, 0)
    if n_open != n_close or len(tags) != n_open:
        return Extraction("malformed_tag", None, len(tags))
    if len(tags) > 1:
        return Extraction("multiple_tags", None, len(tags))
    content = tags[0].strip()
    if not content:
        return Extraction("empty", None, 1)
    if _ALT_SEP_RE.search(content.lower()) or len(_NUM_TOKEN_RE.findall(content)) >= 2:
        return Extraction("multiple_answers", content, 1)
    return Extraction(OK, content, 1)


@dataclass(frozen=True)
class NormAnswer:
    kind: str                   # "number" | "text"
    text: str                   # normalized surface text
    value: Decimal | None = None
    percent: bool = False
    unit: str = ""
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def key(self) -> str:
        """Label-free canonical key used for answer agreement / diversity (signal A)."""
        if self.kind == "number":
            v = self.value.normalize()
            v = v.quantize(Decimal(1)) if v == v.to_integral() else v  # avoid "1E+2"
            return f"num:{v}{'%' if self.percent else ''}{('|' + self.unit) if self.unit else ''}"
        return f"text:{self.text}"


def _basic(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("−", "-").lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip("\"'`")
    s = re.sub(r"\.$", "", s).strip()
    return s


def normalize(s: str) -> NormAnswer:
    t = _basic(s)
    m = _NUMERIC_RE.match(t)
    if m:
        try:
            value = Decimal(m.group("num").replace(",", "").replace(" ", ""))
        except InvalidOperation:
            value = None
        if value is not None:
            if m.group("sign") == "-":
                value = -value
            unit = re.sub(r"\s+", " ", (m.group("unit") or "").strip().rstrip("."))
            return NormAnswer("number", t, value, bool(m.group("pct")), unit)
    return NormAnswer("text", t)


def _match_one(pred: NormAnswer, gold: NormAnswer, question: str) -> bool:
    if pred.kind != gold.kind:
        return False
    if pred.kind == "text":
        return pred.text == gold.text
    if pred.value != gold.value:
        return False
    if pred.percent != gold.percent and not _PERCENT_QUESTION_RE.search(question or ""):
        return False
    if pred.unit and gold.unit and pred.unit != gold.unit:
        return False
    return True


def match_gold(answer: str, labels: list[str], question: str = "") -> bool:
    pred = normalize(answer)
    return any(_match_one(pred, normalize(g), question) for g in labels)


def grade(text: str, labels: list[str], question: str, truncated: bool) -> dict:
    """Extraction + gold correctness, kept separate from any judge verdict."""
    ex = extract_answer(text, truncated)
    correct = ex.status == OK and match_gold(ex.answer, labels, question)
    return {
        "parse_status": ex.status,
        "final_answer": ex.answer if ex.status == OK else None,
        "answer_key": normalize(ex.answer).key if ex.status == OK else None,
        "gold_correct": bool(correct),
        # Why the gold check failed: parse failure vs a wrong extracted answer.
        "gold_fail_reason": None if correct else ("parse_fail" if ex.status != OK else "wrong_answer"),
        "gold_rule": RULE_NAME,
    }
