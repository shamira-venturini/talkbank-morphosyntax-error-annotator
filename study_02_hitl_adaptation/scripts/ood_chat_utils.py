from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence


TIME_MARK_RE = re.compile(r"\x15[^\x15]*\x15")
MAIN_TIER_RE = re.compile(r"^\*([^:]+):\s*(.*)$")


def parse_participants_line(line: str) -> List[str]:
    if not line.startswith("@Participants:"):
        return []
    body = line.split(":", 1)[1].strip()
    if not body:
        return []
    speakers: List[str] = []
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        speakers.append(item.split()[0])
    return speakers


def clean_chat_utterance(text: str) -> str:
    text = TIME_MARK_RE.sub("", text)
    text = text.replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_chat_file(path: Path) -> Dict:
    participants: List[str] = []
    utterances: List[Dict] = []
    pending_idx = -1

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if line.startswith("@Participants:"):
                participants = parse_participants_line(line)
                continue

            if line.startswith("%"):
                # Linguistic analysis tiers are ignored for OOD annotation input.
                continue

            match = MAIN_TIER_RE.match(line)
            if match:
                speaker = match.group(1).strip()
                text = clean_chat_utterance(match.group(2))
                pending_idx += 1
                utterances.append(
                    {
                        "speaker": speaker,
                        "text": text,
                        "line_no": lineno,
                        "raw_index": pending_idx,
                    }
                )
                continue

            if line.startswith("\t") and utterances:
                cont = clean_chat_utterance(line)
                if cont:
                    prev = utterances[-1]["text"]
                    utterances[-1]["text"] = clean_chat_utterance(f"{prev} {cont}")

    return {
        "file_path": str(path),
        "file_name": path.name,
        "participants": participants,
        "utterances": utterances,
    }


def dominant_speaker(utterances: Sequence[Dict]) -> str:
    counts = Counter(utt["speaker"] for utt in utterances)
    if not counts:
        return ""
    # Stable tie-break by lexical order so runs are deterministic.
    best_count = max(counts.values())
    tied = sorted([speaker for speaker, n in counts.items() if n == best_count])
    return tied[0]


def select_speakers(
    utterances: Sequence[Dict],
    participants: Sequence[str],
    policy: str,
    include_speakers: Sequence[str],
) -> List[str]:
    if include_speakers:
        include_set = {s.strip() for s in include_speakers if s.strip()}
        return sorted(include_set)

    if policy == "all":
        return sorted({utt["speaker"] for utt in utterances})

    if policy == "first_participant":
        if participants:
            return [participants[0]]
        dom = dominant_speaker(utterances)
        return [dom] if dom else []

    if policy == "dominant":
        dom = dominant_speaker(utterances)
        return [dom] if dom else []

    raise ValueError(f"Unknown speaker policy: {policy}")
