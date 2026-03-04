import re
import json

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s)

    words = s.split()
    words = [w for w in words if w not in STOPWORDS]

    return " ".join(words)


STOPWORDS = {
    "of","a","an","the","how","can","i","to","please","me","do","did","is","are"
}


SYNONYMS = {
    "solve": ["solve", "solution", "solving"],
    "system": ["system", "systems"],
}

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

# Note: this is a very simple parser that relies on exact matches and regex patterns. It can be easily extended with more sophisticated NLP techniques if needed, but this should work reasonably well for simple queries.
def expand_synonyms(text: str) -> str:
    words = text.split()
    expanded = []
    for w in words:
        expanded.extend(SYNONYMS.get(w, [w]))
    return " ".join(expanded)



def parse_nl_to_ur(text: str, lexicon: dict) -> dict:
    t = expand_synonyms(_norm(text))
    raw = re.sub(r"\s+", " ", text.lower().strip())
    ur = {}

    # topic / subtopic / keyword (auto-match)
    t = expand_synonyms(_norm(text))
    t_tokens = set(t.split())

    for attr in ["topic_name", "subtopic_name", "keyword_name"]:
        phrases = lexicon.get(attr, [])
        matched = []
        for p in phrases:
            p_exp = expand_synonyms(_norm(p))
            if set(p_exp.split()).issubset(t_tokens):
                matched.append(p)
        if matched:
            ur[attr] = list(dict.fromkeys(matched))

    # levels -> newLevel
    levels = []
    for pat in LEVEL_PATTERNS:
        levels += [int(x) for x in re.findall(pat, raw)]
    if levels:
        ur["newLevel"] = sorted(set(levels))

    # question_id
    qids = []
    for pat in QUESTION_PATTERNS:
        qids += [int(x) for x in re.findall(pat, raw)]
    if qids:
        ur["question_id"] = sorted(set(qids))

    # answer1 (ONLY if explicitly mentioned)
    m = re.search(ANSWER_PATTERN, raw)
    if m:
        answer_text = _norm(m.group(1).strip())
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

    query = "How can I solve a linear system 4x4?"
    ur = parse_nl_to_ur(query, LEXICON)

    print(json.dumps(ur, indent=2))