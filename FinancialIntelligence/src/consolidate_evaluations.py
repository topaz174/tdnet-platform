import json, collections, pathlib, re

SRC_DIR = pathlib.Path("evaluation_reports")        # folder with 100 *.json files
DEST    = pathlib.Path("consolidated_qa_report.json")

# ---------- accumulate ----------
totals = collections.Counter()             # category   → count
details = {}                               # category   → Counter(detail)
examples = {}                              # category   → {detail: [(file, idx), …]}

for jf in SRC_DIR.glob("*.json"):
    data = json.loads(jf.read_text(encoding="utf-8"))
    for issue in data.get("issues", []):
        cat   = issue["category"]
        det   = re.sub(r"\s+", " ", issue["detail"]).strip()   # normalise spaces
        chks  = issue.get("example_chunks", [])
        fkey  = jf.stem

        totals[cat] += 1
        details.setdefault(cat, collections.Counter())[det] += 1
        examples.setdefault(cat, {}).setdefault(det, []).extend(
            [(fkey, c) for c in chks][:5]      # cap per file
        )

# ---------- shape output ----------
out = {
    "files_scanned": len(list(SRC_DIR.glob("*.json"))),
    "issue_summary": []
}
for cat, n in totals.most_common():
    det_counter = details[cat]
    cat_block = {
        "category": cat,
        "count": n,
        "top_details": []
    }
    for det, cnt in det_counter.most_common(3):       # top-3 per category
        cat_block["top_details"].append({
            "detail": det,
            "count": cnt,
            "examples": examples[cat][det][:5]        # up to 5 (file, chunk) pairs
        })
    out["issue_summary"].append(cat_block)

DEST.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"Wrote {DEST}")
