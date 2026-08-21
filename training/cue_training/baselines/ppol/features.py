"""19-dimensional behavioral fingerprint of a user's turns (PPol).

These stylometric/behavioral features describe *how* a user communicates and are
used both as the RandomForest discriminator's input and as the descriptor space
for the MAP-Elites archive.
"""

from __future__ import annotations

import re

FEATURE_NAMES = [
    "mean_chars_per_turn",
    "mean_words_per_turn",
    "std_words_per_turn",
    "type_token_ratio",
    "question_ratio",
    "exclamation_ratio",
    "uppercase_char_ratio",
    "digit_char_ratio",
    "punctuation_ratio",
    "mean_sentence_len",
    "politeness_rate",
    "imperative_start_ratio",
    "first_person_rate",
    "second_person_rate",
    "filler_rate",
    "emoji_rate",
    "typo_proxy_rate",
    "code_token_rate",
    "user_turn_count",
]
N_FEATURES = len(FEATURE_NAMES)

_WORD_RE = re.compile(r"[A-Za-z']+")
_CLEAN_WORD_RE = re.compile(r"^[a-z][a-z']*$")
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]"
)
_POLITE = {"please", "thanks", "thank", "sorry", "appreciate", "kindly"}
_FILLER = {"um", "uh", "like", "well", "hmm", "yeah", "ok", "okay"}
_FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}
_SECOND_PERSON = {"you", "your", "yours"}
_IMPERATIVE_STARTS = {
    "add", "fix", "make", "write", "create", "remove", "change", "update",
    "show", "give", "tell", "run", "build", "use", "set", "put", "explain",
    "help", "do", "check", "find", "list", "generate", "implement",
}


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def user_turns(conversation: list[dict[str, str]]) -> list[str]:
    return [
        str(t.get("content", ""))
        for t in conversation
        if t.get("role") == "user" and t.get("content")
    ]


def fingerprint(conversation: list[dict[str, str]]) -> list[float]:
    """Compute the 19-feature fingerprint for a conversation's user turns."""

    turns = user_turns(conversation)
    if not turns:
        return [0.0] * N_FEATURES

    char_counts, word_counts = [], []
    all_words: list[str] = []
    n_q = n_excl = n_imp = 0
    up_chars = digit_chars = punct_chars = total_chars = 0
    sentences = 0
    polite = filler = first_p = second_p = emoji = typo = code = 0

    for turn in turns:
        char_counts.append(len(turn))
        words = _WORD_RE.findall(turn)
        lowered = [w.lower() for w in words]
        word_counts.append(len(words))
        all_words.extend(lowered)
        total_chars += len(turn)
        up_chars += sum(1 for c in turn if c.isupper())
        digit_chars += sum(1 for c in turn if c.isdigit())
        punct_chars += sum(1 for c in turn if c in ".,!?;:-_()[]{}\"'")
        sentences += max(1, len(re.split(r"[.!?]+", turn.strip())) - 1) or 1
        if "?" in turn:
            n_q += 1
        if "!" in turn:
            n_excl += 1
        if lowered and lowered[0] in _IMPERATIVE_STARTS:
            n_imp += 1
        polite += sum(1 for w in lowered if w in _POLITE)
        filler += sum(1 for w in lowered if w in _FILLER)
        first_p += sum(1 for w in lowered if w in _FIRST_PERSON)
        second_p += sum(1 for w in lowered if w in _SECOND_PERSON)
        emoji += len(_EMOJI_RE.findall(turn))
        typo += sum(1 for w in lowered if not _CLEAN_WORD_RE.match(w))
        if any(tok in turn for tok in ("`", "()", "{", "};", "def ", "import ")):
            code += 1

    n = len(turns)
    total_words = sum(word_counts) or 1
    mean_words = _safe_div(sum(word_counts), n)
    var = _safe_div(sum((w - mean_words) ** 2 for w in word_counts), n)
    ttr = _safe_div(len(set(all_words)), len(all_words))

    return [
        _safe_div(sum(char_counts), n),
        mean_words,
        var ** 0.5,
        ttr,
        _safe_div(n_q, n),
        _safe_div(n_excl, n),
        _safe_div(up_chars, total_chars),
        _safe_div(digit_chars, total_chars),
        _safe_div(punct_chars, total_chars),
        _safe_div(total_words, sentences),
        _safe_div(polite, n),
        _safe_div(n_imp, n),
        _safe_div(first_p, total_words),
        _safe_div(second_p, total_words),
        _safe_div(filler, total_words),
        _safe_div(emoji, n),
        _safe_div(typo, total_words),
        _safe_div(code, n),
        float(n),
    ]
