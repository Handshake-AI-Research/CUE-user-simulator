"""Deterministic gates for data annotation persona manuals."""

from __future__ import annotations

import re
from typing import Any

# Soft/generic phrases that make human_contrast uninformative.
_GENERIC_HUMAN = re.compile(
    r"\b("
    r"be cooperative|be polite|stay calm|matter[- ]of[- ]fact|brief courtesy|"
    r"politely|overly polished|casual phrasing|short,? direct|"
    r"task[- ]focused|without adding|acknowledge it briefly"
    r")\b",
    re.I,
)

# Content / task nouns that should not appear in manuals (after light delex).
_CONTENT_LEAK = re.compile(
    r"\b("
    r"shirt|password|pin|username|order\s*id|account\s*id|refund|return\s*policy|"
    r"shipping\s*label|membership|bronze|silver|gold|guest|receipt|package|"
    r"guess\s*shirt|email\s*address|mailing\s*address"
    r")\b",
    re.I,
)

# Commands that claim emotional reactions we can spot-check against the transcript.
_FRUSTRATION_CLAIM = re.compile(r"\b(frustrat|disappoint|angry|upset|annoy|unfair)\w*\b", re.I)
_ACCEPTANCE_IN_TRANSCRIPT = re.compile(
    r"\b(ok(ay)?|i understand|its okay|it's okay|that's fine|no problem|thanks?)\b",
    re.I,
)

# Manual must mention lag when the transcript shows it (broad synonyms).
_LAG_CAPTURE = re.compile(
    r"("
    r"\blag(?:ged|ging)?\b|"
    r"\bout[- ]of[- ]order\b|"
    r"\bone (?:question|turn|prompt) behind\b|"
    r"\b(?:previous|prior|earlier|older)\s+(?:question|prompt|ask|thread|turn)\b|"
    r"\b(?:previous|prior|earlier|older)\s+\w+\s+(?:question|prompt|ask|thread)\b|"
    r"\bstagger(?:ed|ing)?\b|"
    r"\bdelay(?:ed|ing)?\s+(?:answer|reply|response|turn)\b|"
    r"\b(?:answer|reply|respond)(?:s|ing|ed)?\s+(?:to\s+)?(?:the\s+)?"
    r"(?:previous|prior|earlier|older|last)\b|"
    r"\breply(?:ing)?\s+to\s+(?:the\s+)?(?:previous|prior|earlier|older)\b|"
    r"\bbehind\s+(?:the\s+)?(?:assistant|agent|latest)\b|"
    r"\bnot\s+(?:the\s+)?(?:most\s+)?(?:recent|latest)\s+(?:question|prompt|one)\b|"
    r"\brather than\s+(?:the\s+)?(?:latest|newest|current)\b|"
    r"\binstead of\s+(?:the\s+)?(?:latest|newest|current)\b|"
    r"\balready moved on\b|"
    r"\bolder thread\b|"
    r"\bconversation has already moved\b|"
    r"\btiming pattern\b|"
    r"\b(?:even when|after)\s+(?:the\s+)?(?:conversation|thread)\s+has\s+(?:already\s+)?moved\b"
    r")",
    re.I,
)

# Claims that contradict observed answer lag.
_SEQUENTIAL_CLAIM = re.compile(
    r"("
    r"\banswer(?:s|ing)?\s+(?:the\s+)?(?:current|immediate|latest|present)\s+(?:question|prompt)\b|"
    r"\brespond(?:s|ing)?\s+directly\s+to\s+the\s+current\b|"
    r"\bwait for the next prompt\b|"
    r"\bback[- ]and[- ]forth turn pattern\b|"
    r"\bstep[- ]by[- ]step progression\b|"
    r"\bprovide .{0,40}in the next turn rather than\b|"
    r"\banswer .{0,40}(?:when|as soon as) asked\b|"
    r"\bfollow(?:s|ing)?\s+the\s+assistant'?s\s+sequence\b|"
    r"\bsteady,?\s+cooperative\s+back[- ]and[- ]forth\b|"
    r"\b(?:do not|don't|without)\s+(?:answer|reply|respond).{0,40}out[- ]of[- ]order\b|"
    r"\b(?:same|strict)\s+turn\s+sequence\b|"
    r"\bwithout .{0,60}out[- ]of[- ]order\b|"
    r"\baligned to the immediately relevant prompt\b|"
    r"\bcompact and sequential\b|"
    r"\banswer only the specific prompt\b|"
    r"\bin order(?: rather than|,\s*not)\b|"
    r"\bperfectly sequential\b|"
    r"\bstay(?:ing)? perfectly sequential\b"
    r")",
    re.I,
)

_ASK_TYPES: list[tuple[str, re.Pattern[str]]] = [
    ("name", re.compile(r"\b(full name|your name|name please|what(?:'s| is) your name)\b", re.I)),
    ("username", re.compile(r"\buser\s*name\b", re.I)),
    ("email", re.compile(r"\bemail\b", re.I)),
    ("order_id", re.compile(r"\border\s*id\b", re.I)),
    ("account_id", re.compile(r"\baccount\s*(?:id|number)\b", re.I)),
    ("membership", re.compile(r"\bmembership\b", re.I)),
    ("date", re.compile(r"\b(purchase date|what day|when .{0,20}(?:purchas|order|bought))\b", re.I)),
    ("phone", re.compile(r"\b(?:phone(?:\s*number)?|mobile)\b", re.I)),
    ("zip", re.compile(r"\bzip(?:\s*code)?\b", re.I)),
    ("pin", re.compile(r"\bpin\b", re.I)),
    ("security", re.compile(r"\bsecurity (?:question|answer)\b", re.I)),
    ("status", re.compile(r"\b(?:shipping )?status|in transit|out for delivery\b", re.I)),
    ("yes_no", re.compile(
        r"\b(?:is that correct|does that|right\?|you've already|you want to|already received)\b",
        re.I,
    )),
    ("closing", re.compile(r"\b(?:anything else|is there anything else|will this be all)\b", re.I)),
]


def _classify_answers(text: str) -> set[str]:
    """High-precision answer-slot tags (avoid matching ordinary English words)."""

    types: set[str] = set()
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        types.add("email")
    if re.search(r"(?:\+?\d{1,2}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", text):
        types.add("phone")
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", text):
        types.add("date")
    if re.search(r"\b(?:bronze|silver|gold|platinum|guest)\b", text, re.I):
        types.add("membership")
    if re.search(r"\border\s*id\b|\b\d{7,}\b", text, re.I):
        types.add("order_id")
    if re.search(r"\baccount\s*id\b|\b[A-Z]{2,}[A-Z0-9]{6,}\b", text, re.I):
        types.add("account_id")
    if re.search(r"\buser\s*name\s*[:#]?\s*\S+", text, re.I):
        types.add("username")
    if re.fullmatch(r"\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\s*", text):
        types.add("name")
    if re.search(r"\b\d{5}(?:-\d{4})?\b", text) and re.search(r"\b(?:zip|phone|\(\d{3}\))", text, re.I):
        types.add("zip")
    elif re.fullmatch(r"\s*\d{5}(?:-\d{4})?\s*", text):
        types.add("zip")
    if re.fullmatch(r"\s*\d{4,8}\s*", text):
        types.add("pin")
    if re.search(r"\b(?:delivered|in transit|out for delivery|shipped)\b", text, re.I):
        types.add("status")
    if re.match(r"^\s*(?:yes|no|yeah|yep|nope)\b", text, re.I) or re.search(
        r"\b(?:that is correct|that's correct)\b", text, re.I
    ):
        types.add("yes_no")
    if re.search(r"\b(?:security )?answer\s*(?:is|:)?\s*[A-Za-z]{3,20}\b", text, re.I):
        types.add("security")
    return types


class ManualGateError(ValueError):
    """Raised when a manual fails deterministic quality gates."""


# Session signatures used for human_contrast distinctiveness vs negatives.
_SIG_ESCALATION_TX = re.compile(r"\b(?:manager|supervisor|escalat)\w*\b", re.I)
_SIG_PUSHBACK_TX = re.compile(
    r"\b(?:exception|what do you mean|totally an issue|got to be kidding|that'?s totally)\b",
    re.I,
)
_SIG_IMPATIENCE_TX = re.compile(
    r"\b(?:ain'?t got all day|hurry|all day holmes|makin me)\b",
    re.I,
)
_SIG_TRY_REPORT_TX = re.compile(
    r"\b(?:tried that|no dice|still (?:seems|doing)|that (?:worked|did the trick)|"
    r"give me a second|just a sec)\b",
    re.I,
)
_SIG_SMALLTALK_TX = re.compile(
    r"\b(?:how are you|weather|how long have you been with)\b",
    re.I,
)
_SIG_PLAYFUL_TX = re.compile(r"\b(?:owo|type owo)\b|[A-Z]{10,}", re.I)
_SIG_ACCEPT_TX = re.compile(
    r"\b(?:ok(?:ay)?(?:[,.]?\s*i understand)?|i understand|its okay|it's okay|that is okay|"
    r"thanks for trying)\b",
    re.I,
)
_SIG_REFUSAL_ASST = re.compile(
    r"\b(?:cannot|can'?t|unable|not (?:able|allowed)|out of the (?:return )?period|"
    r"i do not have the authority|nothing (?:i|we) can do)\b",
    re.I,
)

_CLAIM_BY_SIG: dict[str, re.Pattern[str]] = {
    "answer_lag": _LAG_CAPTURE,
    "escalation": re.compile(r"\b(?:manager|escalat\w*|supervisor|handoff|hand[- ]off)\b", re.I),
    "pushback": re.compile(
        r"\b(?:push\s*back|exception|correct(?:ion|ing)?|challeng\w*|dispute|insist|"
        r"do not (?:accept|immediately accept))\b",
        re.I,
    ),
    "impatience": re.compile(
        r"\b(?:impatient|hurry|hurried|compressed style|ain'?t got|aint got)\b",
        re.I,
    ),
    "try_report": re.compile(
        r"\b(?:try[- ]?(?:then[- ]?)?report|tried .{0,40}(?:fail|work|result)|"
        r"report(?:s|ing)? (?:the )?result|status update after)\b",
        re.I,
    ),
    "smalltalk": re.compile(
        r"\b(?:small[- ]talk|side[- ]topic|digression|weather|how are you)\b",
        re.I,
    ),
    "playful": re.compile(
        r"\b(?:playful|meme|owo|internet[- ]style|keyboard|noisy outburst)\b",
        re.I,
    ),
    "acceptance": re.compile(
        r"\b(?:accept(?:s|ing|ance)? .{0,40}(?:refus|limit|cannot|no\b|negative)|"
        r"do not escalate|without (?:arguing|pushing|escalat)|"
        r"acknowledge (?:it|the (?:limit|refusal|outcome))|"
        r"move toward closure rather than argu)\b",
        re.I,
    ),
}


def _user_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [t for t in turns if t.get("role") == "user"]


def _user_blob(turns: list[dict[str, Any]]) -> str:
    return "\n".join(str(t.get("content") or "") for t in _user_turns(turns))


def _classify(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> set[str]:
    return {name for name, pat in patterns if pat.search(text)}


def detect_answer_lag(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find user turns that answer an earlier assistant ask instead of the latest one."""

    ask_history: list[tuple[set[str], str]] = []
    events: list[dict[str, Any]] = []
    for turn in turns:
        role = turn.get("role")
        content = str(turn.get("content") or "")
        if role == "assistant":
            asks = _classify(content, _ASK_TYPES)
            if asks:
                ask_history.append((asks, content[:100]))
            continue
        if role != "user" or len(ask_history) < 2:
            continue
        answers = _classify_answers(content)
        if not answers and re.fullmatch(r"\s*[A-Za-z]{3,20}\s*", content):
            # Lone token: treat as security/name only if those asks are pending.
            pending = set().union(*(atypes for atypes, _ in ask_history[-3:]))
            if "security" in pending:
                answers.add("security")
            elif "name" in pending:
                answers.add("name")
        if not answers:
            continue
        latest_types, latest_snip = ask_history[-1]
        score_latest = len(answers & latest_types)
        best_older_i = -1
        best_older_score = 0
        for i, (atypes, _) in enumerate(ask_history[:-1]):
            score = len(answers & atypes)
            if score > best_older_score:
                best_older_score = score
                best_older_i = i
        # Lag / out-of-order: stronger match to an older ask than to the latest ask.
        if best_older_score > score_latest and best_older_score > 0:
            older_types, older_snip = ask_history[best_older_i]
            events.append(
                {
                    "user_turn_id": str(turn.get("turn_id") or ""),
                    "answer_types": sorted(answers),
                    "older_ask_types": sorted(older_types),
                    "latest_ask_types": sorted(latest_types),
                    "older_ask": older_snip,
                    "latest_ask": latest_snip,
                }
            )
    return events


def detect_session_signatures(turns: list[dict[str, Any]]) -> set[str]:
    """Coarse behavioral signatures present in a session (for contrast checks)."""

    sigs: set[str] = set()
    user_blob = _user_blob(turns)
    asst_blob = "\n".join(
        str(t.get("content") or "") for t in turns if t.get("role") == "assistant"
    )
    if len(detect_answer_lag(turns)) >= 2:
        sigs.add("answer_lag")
    if _SIG_ESCALATION_TX.search(user_blob):
        sigs.add("escalation")
    if _SIG_PUSHBACK_TX.search(user_blob):
        sigs.add("pushback")
    if _SIG_IMPATIENCE_TX.search(user_blob):
        sigs.add("impatience")
    if _SIG_TRY_REPORT_TX.search(user_blob):
        sigs.add("try_report")
    if _SIG_SMALLTALK_TX.search(user_blob):
        sigs.add("smalltalk")
    if _SIG_PLAYFUL_TX.search(user_blob):
        sigs.add("playful")
    if _SIG_REFUSAL_ASST.search(asst_blob) and _SIG_ACCEPT_TX.search(user_blob):
        sigs.add("acceptance")
    return sigs


def signatures_claimed_by_command(cmd: dict[str, Any]) -> set[str]:
    blob = f"{cmd.get('text')} {' '.join(cmd.get('examples') or [])}"
    return {name for name, pat in _CLAIM_BY_SIG.items() if pat.search(blob)}


def distinctive_human_commands(
    human_cmds: list[dict[str, Any]],
    *,
    target_turns: list[dict[str, Any]],
    negative_turns: list[list[dict[str, Any]]],
    max_neg_hits: int = 1,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Human commands that claim a TARGET signature rare among negatives."""

    target_sigs = detect_session_signatures(target_turns)
    neg_counts: dict[str, int] = {s: 0 for s in target_sigs}
    for neg in negative_turns:
        neg_sigs = detect_session_signatures(neg)
        for s in target_sigs:
            if s in neg_sigs:
                neg_counts[s] += 1
    available = {s for s, n in neg_counts.items() if n <= max_neg_hits}
    distinctive: list[dict[str, Any]] = []
    for cmd in human_cmds:
        claimed = signatures_claimed_by_command(cmd) & available
        if claimed:
            distinctive.append(cmd)
    return distinctive, available


def ground_evidence_turn_ids(
    manual: dict[str, Any],
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fill missing evidence_turn_ids from overlap with user turns; drop invalid ids."""

    users = _user_turns(turns)
    valid_ids = {str(t.get("turn_id")) for t in users if t.get("turn_id")}
    commands = []
    for cmd in manual.get("commands") or []:
        evidence = [str(x) for x in (cmd.get("evidence_turn_ids") or []) if str(x) in valid_ids]
        if not evidence and users:
            text = str(cmd.get("text") or "").lower()
            examples = " ".join(str(x) for x in (cmd.get("examples") or [])).lower()
            scored: list[tuple[int, str]] = []
            for turn in users:
                content = str(turn.get("content") or "").lower()
                tid = str(turn.get("turn_id") or "")
                if not tid or not content:
                    continue
                score = 0
                for token in re.findall(r"[a-z']{4,}", text + " " + examples):
                    if token in content:
                        score += 1
                if score:
                    scored.append((score, tid))
            scored.sort(reverse=True)
            evidence = [tid for _, tid in scored[:2]]
            if not evidence:
                # Fall back to first + last user turns so the field is never empty.
                evidence = [str(users[0].get("turn_id"))]
                if len(users) > 1:
                    evidence.append(str(users[-1].get("turn_id")))
        commands.append({**cmd, "evidence_turn_ids": evidence})
    return {**manual, "commands": commands}


def gate_manual(
    manual: dict[str, Any],
    *,
    turns: list[dict[str, Any]],
    negatives: list[list[dict[str, Any]]] | None = None,
    min_distinctive_human: int = 2,
) -> None:
    """Raise ManualGateError if the manual fails quality checks."""

    commands = list(manual.get("commands") or [])
    if len(commands) < 8:
        raise ManualGateError(f"too few commands: {len(commands)}")

    sim = [c for c in commands if c.get("kind") == "sim_contrast"]
    human = [c for c in commands if c.get("kind") == "human_contrast"]
    if len(sim) < 4 or len(human) < 4:
        raise ManualGateError(f"need >=4 sim and >=4 human commands; got {len(sim)}/{len(human)}")

    missing_ev = [c for c in commands if not c.get("evidence_turn_ids")]
    if missing_ev:
        raise ManualGateError(f"{len(missing_ev)} commands missing evidence_turn_ids")

    leaks = []
    for c in commands:
        blob = f"{c.get('text')} {' '.join(c.get('examples') or [])}"
        if _CONTENT_LEAK.search(blob):
            leaks.append(str(c.get("text") or "")[:80])
    if leaks:
        raise ManualGateError("content leak in commands: " + " | ".join(leaks[:3]))

    generic_human = sum(1 for c in human if _GENERIC_HUMAN.search(str(c.get("text") or "")))
    if generic_human >= 3:
        raise ManualGateError(
            f"human_contrast too generic ({generic_human}/ {len(human)} soft templates)"
        )

    user_text = _user_blob(turns)
    for c in commands:
        text = str(c.get("text") or "")
        if _FRUSTRATION_CLAIM.search(text) and _ACCEPTANCE_IN_TRANSCRIPT.search(user_text):
            # Only flag when the user looks accepting and never frustrated.
            if not _FRUSTRATION_CLAIM.search(user_text):
                raise ManualGateError(
                    "command claims frustration/disappointment but transcript is accepting"
                )

    lag_events = detect_answer_lag(turns)
    if lag_events:
        cmd_blob = " ".join(
            f"{c.get('text')} {' '.join(c.get('examples') or [])}" for c in commands
        )
        # Any lag + sequential claim is a hard contradiction.
        for c in commands:
            claim_blob = f"{c.get('text')} {' '.join(c.get('examples') or [])}"
            if _SEQUENTIAL_CLAIM.search(claim_blob):
                raise ManualGateError(
                    "command claims sequential/on-prompt answering but transcript shows answer lag"
                )
        # Repeated lag must be encoded as a turn-timing behavior.
        if len(lag_events) >= 2 and not _LAG_CAPTURE.search(cmd_blob):
            ids = [e["user_turn_id"] for e in lag_events if e.get("user_turn_id")]
            raise ManualGateError(
                f"transcript has {len(lag_events)} answer-lag/out-of-order replies "
                f"(e.g. {', '.join(ids[:3])}) but manual never mentions lag / "
                "out-of-order / previous-question answering"
            )

    # Require some human_contrast commands that are true of TARGET and rare in negatives.
    if negatives:
        distinctive, available = distinctive_human_commands(
            human,
            target_turns=turns,
            negative_turns=negatives,
        )
        required = min(min_distinctive_human, len(available))
        if required and len(distinctive) < required:
            raise ManualGateError(
                f"need >={required} human_contrast commands that encode TARGET-only "
                f"signatures {sorted(available)}; got {len(distinctive)} "
                "(others are shared with human negatives or only soft closers)"
            )
