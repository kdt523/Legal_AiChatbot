"""
generate_test_dataset.py — Create 150 high-quality, realistic legal test queries.

Queries are sampled from across the different statutes (IPC, CrPC, CPC, IEA, MVA, NI Act, IDA)
and categorized into:
  - direct: "What is Section 302 IPC?"
  - conceptual: "Explain the definition of murder."
  - punishment: "What is the penalty for theft?"
  - procedural: "How is an arrest made under CrPC?"
  - edge: Tricky combinations or confusing titles.
"""

import json
import random
import pandas as pd
from pathlib import Path

# Load law data to ensure we pick real sections
DATA_DIR = Path("data/raw")
FILES = ["ipc.json", "crpc.json", "iea.json", "MVA.json", "nia.json", "cpc.json", "ida.json"]

# IPC Chapter map we built earlier to help with section numbers
IPC_MAP = {
    "1": 1, "2": 14, "3": 52, "4": 53, "5": 76, "6": 121, "7": 141, "8": 153, "9": 161,
    "10": 191, "11": 211, "12": 231, "13": 264, "14": 268, "15": 295, "16": 299,
    "17": 378, "18": 415, "19": 463, "20": 494, "21": 499, "22": 503, "23": 511
}

def get_real_sections():
    all_sections = []
    for f in FILES:
        path = DATA_DIR / f
        if not path.exists(): continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
            for i, entry in enumerate(data):
                if not isinstance(entry, dict) or "section_title" not in entry: continue
                # Infer section number for IPC if missing
                snum = str(entry.get("section", "")).strip()
                if not snum and f == "ipc.json":
                    ch = str(entry.get("chapter", ""))
                    if ch in IPC_MAP:
                        # This is a bit of an approximation but helps for test generation
                        # We'll just use the title mostly and manual overrides for key sections
                        pass
                all_sections.append({
                    "src": f,
                    "title": entry["section_title"],
                    "desc": entry.get("section_desc", ""),
                    "chapter": entry.get("chapter_title", ""),
                    "raw_idx": i
                })
    return all_sections

def generate_questions():
    sections = get_real_sections()
    dataset = []

    # 1. Manual Golden Set (The "Resume" Questions)
    golden = [
        ("What is the punishment for murder under IPC Section 302?", "Punishment for murder", "302", "easy", "punishment"),
        ("What characterizes 'Culpable Homicide' in Indian law?", "Culpable homicide", "299", "medium", "conceptual"),
        ("How can a person be arrested without a warrant under CrPC?", "Arrest by police without warrant", "41", "medium", "procedural"),
        ("What is the punishment for cheque bounce under NI Act?", "Dishonour of cheque", "138", "easy", "punishment"),
        ("When is 'hearsay evidence' admissible under the Evidence Act?", "Evidence of character", "55", "hard", "conceptual"),
        ("What is the penalty for driving without a license under MVA?", "Driving without license", "181", "easy", "punishment"),
        ("Define 'Grevious Hurt' under the IPC.", "Grievous hurt", "320", "medium", "conceptual"),
        ("What is the procedure for bail in non-bailable offences?", "When bail may be taken in case of non-bailable offence", "437", "hard", "procedural"),
        ("What is the definition of theft?", "Theft", "378", "easy", "conceptual"),
        ("What constitutes 'Criminal Breach of Trust'?", "Criminal breach of trust", "405", "medium", "conceptual"),
    ]
    for q, law, s, diff, qtype in golden:
        dataset.append({"question": q, "expected_law": law, "expected_section": s, "difficulty_level": diff, "query_type": qtype})

    # 2. Automated Generation (using templates)
    templates = {
        "punishment": [
            "What is the punishment for {}?",
            "What is the penalty for {} under the law?",
            "How is the offence of {} punished?",
        ],
        "conceptual": [
            "Explain the concept of {} in Indian law.",
            "What defines the provision of {}?",
            "Define {} as per the statutes.",
        ],
        "direct": [
            "What does Section {} of {} deal with?",
            "Explain Section {} {}.",
        ]
    }

    # Sample a diverse set from each law
    by_law = {}
    for s in sections:
        by_law.setdefault(s['src'], []).append(s)

    # Need to reach 150 total. We have 10 golden. Need 140 more.
    # Target counts: IPC (40), CrPC (30), IEA (20), MVA (20), NIA (10), CPC (10), IDA (10) = 140
    targets = {"ipc.json": 40, "crpc.json": 30, "iea.json": 20, "MVA.json": 20, "nia.json": 10, "cpc.json": 10, "ida.json": 10}

    for law_file, count in targets.items():
        sub_sections = by_law.get(law_file, [])
        if not sub_sections: continue
        sampled = random.sample(sub_sections, min(count, len(sub_sections)))

        for s in sampled:
            title = s['title']
            qtype = random.choice(["punishment", "conceptual", "direct"])
            diff = random.choice(["easy", "medium", "hard"])

            if qtype == "direct":
                # We'll use the law name for direct lookups
                lname = law_file.replace(".json", "").upper()
                # For direct we need a numeric section. If title has it use it, else skip
                num_match = "".join(filter(str.isdigit, title))
                if num_match:
                    q = random.choice(templates["direct"]).format(num_match, lname)
                    snum = num_match
                else:
                    # Fallback to conceptual if no number
                    q = random.choice(templates["conceptual"]).format(title)
                    snum = ""
            else:
                q = random.choice(templates[qtype]).format(title)
                snum = ""

            dataset.append({
                "question": q,
                "expected_law": title,
                "expected_section": snum,
                "difficulty_level": diff,
                "query_type": qtype
            })

    # Fill up to 150 if needed
    while len(dataset) < 150:
        s = random.choice(sections)
        dataset.append({
            "question": f"Tell me about {s['title']} in {s['src']}.",
            "expected_law": s['title'],
            "expected_section": "",
            "difficulty_level": "medium",
            "query_type": "conceptual"
        })

    return dataset[:150]

if __name__ == "__main__":
    print("Generating 150 legal queries...")
    data = generate_questions()
    df = pd.DataFrame(data)
    df.to_csv("data/test_questions.csv", index=False)
    print(f"Generated data/test_questions.csv with {len(df)} rows.")
