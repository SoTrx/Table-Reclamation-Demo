import re
import json

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

LEVEL_PATTERNS = [
    r"\blevel\s*(\d+)\b",
    r"\blvl\s*(\d+)\b",
    r"\bgrade\s*(\d+)\b",
]

QUESTION_PATTERNS = [
    r"\bquestion\s*(\d+)\b",
    r"\bq\s*(\d+)\b",
    r"\bid\s*(\d+)\b",
]

ANSWER_PATTERN = r"\banswer\s*(?:is|:)?\s*(.+)"  # explicit only

def parse_nl_to_ur(text: str, lexicon: dict) -> dict:
    t = _norm(text)
    ur = {}

    # topic / subtopic / keyword (auto-match)
    for attr in ["topic_name", "subtopic_name", "keyword_name"]:
        phrases = lexicon.get(attr, [])
        matched = [p for p in phrases if _norm(p) in t]
        if matched:
            ur[attr] = list(dict.fromkeys(matched))

    # levels -> newLevel
    levels = []
    for pat in LEVEL_PATTERNS:
        levels += [int(x) for x in re.findall(pat, t)]
    if levels:
        ur["newLevel"] = sorted(set(levels))

    # question_id
    qids = []
    for pat in QUESTION_PATTERNS:
        qids += [int(x) for x in re.findall(pat, t)]
    if qids:
        ur["question_id"] = sorted(set(qids))

    # answer1 (ONLY if explicitly mentioned)
    m = re.search(ANSWER_PATTERN, t)
    if m:
        answer_text = m.group(1).strip()
        # match against lexicon values
        answers = lexicon.get("answer1", [])
        matched_answers = [
            a for a in answers if _norm(a) == answer_text
        ]
        if matched_answers:
            ur["answer1"] = matched_answers

    return ur


# ---- RUN ----
if __name__ == "__main__":
    with open("demo/lexicon.json") as f:
        LEXICON = json.load(f)

    query = "Discrete Mathematics Recursivity level 2"
    ur = parse_nl_to_ur(query, LEXICON)

    print(json.dumps(ur, indent=2))