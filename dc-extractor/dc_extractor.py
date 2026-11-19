"""
Data Centre Project Extractor
-----------------------------
LLM-assisted pipeline that turns unstructured sources (news articles, company
announcements, planning documents) into a structured, validated, auditable
project-level dataset of data centre assets.

Pipeline:  collect -> extract (LLM) -> validate (QC) -> deduplicate -> evaluate -> export

Usage:
    python dc_extractor.py collect   --urls data/urls.txt        # optional: fetch web pages into data/raw/
    python dc_extractor.py extract                                # LLM extraction of every file in data/raw/
    python dc_extractor.py evaluate  --gold data/gold/gold.csv    # field-level accuracy vs hand-labelled sample
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("output")
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5")

STATUSES = ["announced", "planning", "permitted", "under_construction",
            "operational", "cancelled", "unknown"]
FIELDS = ["project_name", "developer", "operator", "country", "city_or_region",
          "capacity_mw", "status", "expected_online_year", "power_supply",
          "investment_usd_m"]
REQUIRED = ["project_name", "country", "status"]

# ---------------------------------------------------------------------------
# 1. COLLECT  (web data collection)
# ---------------------------------------------------------------------------

def collect(urls_file: str) -> None:
    import requests
    from bs4 import BeautifulSoup

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in Path(urls_file).read_text().splitlines() if u.strip()]
    for url in urls:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (research project)"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n")).strip()
            fname = hashlib.md5(url.encode()).hexdigest()[:10] + ".txt"
            (RAW_DIR / fname).write_text(f"SOURCE_URL: {url}\n\n{text}", encoding="utf-8")
            print(f"[ok]   {url} -> {fname}")
        except Exception as e:
            print(f"[fail] {url}: {e}")

# ---------------------------------------------------------------------------
# 2. EXTRACT  (LLM + prompt engineering)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a research analyst building a project-level database of data centre assets.
Extract every distinct data centre PROJECT described in the source text.

Rules:
- Only extract facts explicitly stated in the text. Never guess or use outside knowledge.
- If a field is not stated, use null.
- For every non-null field, give a short verbatim quote from the text as evidence.
- capacity_mw: IT or power capacity in megawatts as a number (convert GW to MW). If only a range, use the upper value and note it.
- status must be one of: announced, planning, permitted, under_construction, operational, cancelled, unknown.
- investment_usd_m: investment in USD millions if stated in USD; otherwise null and mention the currency in notes.
- confidence: your confidence (0-1) that the record is accurate overall.

Return ONLY valid JSON in this shape:
{"projects": [{
  "project_name": str, "developer": str|null, "operator": str|null,
  "country": str|null, "city_or_region": str|null,
  "capacity_mw": number|null, "status": str,
  "expected_online_year": int|null, "power_supply": str|null,
  "investment_usd_m": number|null,
  "evidence": {"<field>": "<verbatim quote>", ...},
  "confidence": number, "notes": str|null
}]}
If the text describes no data centre projects, return {"projects": []}."""


def call_llm(text: str) -> dict:
    """Single place to swap LLM providers (Anthropic shown; OpenAI/Gemini work the same way)."""
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"SOURCE TEXT:\n\n{text[:60000]}"}],
    )
    raw = msg.content[0].text
    match = re.search(r"\{.*\}", raw, re.S)
    return json.loads(match.group(0)) if match else {"projects": []}


CACHE_DIR = OUT_DIR / "llm_cache"


def extract(use_cache: bool = True) -> pd.DataFrame:
    """LLM responses are cached per source file, so re-runs are free and reproducible."""
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    rows = []
    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        sys.exit("No files in data/raw/. Add .txt sources or run `collect` first.")
    for f in files:
        text = f.read_text(encoding="utf-8")
        url = text.split("\n", 1)[0].replace("SOURCE_URL:", "").strip() if text.startswith("SOURCE_URL:") else ""
        cache_file = CACHE_DIR / (f.stem + ".json")
        try:
            if use_cache and cache_file.exists():
                result = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                result = call_llm(text)
                cache_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"[llm-fail] {f.name}: {e}")
            continue
        for p in result.get("projects", []):
            p["source_file"] = f.name
            p["source_url"] = url
            p["extracted_on"] = date.today().isoformat()
            p["_source_text"] = text
            rows.append(p)
        print(f"[extracted] {f.name}: {len(result.get('projects', []))} project(s)")

    df = pd.DataFrame(rows)
    df = validate(df)
    df = deduplicate(df)
    export(df)
    return df

# ---------------------------------------------------------------------------
# 3. VALIDATE  (quality control so AI output is accurate, sourced, auditable)
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).lower().strip()


def validate(df: pd.DataFrame) -> pd.DataFrame:
    flags_col = []
    for _, r in df.iterrows():
        flags = []
        for f in REQUIRED:
            if pd.isna(r.get(f)) or r.get(f) in ("", None):
                flags.append(f"missing_{f}")
        if r.get("status") not in STATUSES:
            flags.append("invalid_status")
        cap = r.get("capacity_mw")
        if pd.notna(cap) and not (0 < float(cap) <= 10000):
            flags.append("capacity_out_of_range")
        yr = r.get("expected_online_year")
        if pd.notna(yr) and not (2000 <= int(yr) <= 2045):
            flags.append("year_out_of_range")
        # Hallucination check: every evidence quote must appear in the source text
        src = _norm(r.get("_source_text", ""))
        ev = r.get("evidence") or {}
        for field, quote in ev.items():
            if quote and _norm(quote) not in src:
                flags.append(f"evidence_not_found:{field}")
        # Any populated field without evidence is unsourced
        for f in FIELDS:
            if pd.notna(r.get(f)) and r.get(f) not in ("", None) and f not in ev:
                flags.append(f"unsourced:{f}")
        if not r.get("source_url") and not r.get("source_file"):
            flags.append("no_source")
        if float(r.get("confidence", 0) or 0) < 0.6:
            flags.append("low_confidence")
        flags_col.append(";".join(flags))

    df["qc_flags"] = flags_col
    df["qc_status"] = df["qc_flags"].apply(lambda x: "pass" if not x else "review")
    return df

# ---------------------------------------------------------------------------
# 4. DEDUPLICATE  (same project reported by multiple sources)
# ---------------------------------------------------------------------------

GENERIC = {"data", "centre", "center", "centres", "centers", "campus", "scheme", "project", "facility",
           "ai", "the", "dc", "site", "factory", "hyperscale", "mega", "development", "expansion"}


def _tokens(s) -> set:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", str(s).lower()) if t not in GENERIC and len(t) > 1}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def is_same_project(a, b) -> bool:
    """Two records describe the same asset if they are in the same country and either
    (1) their names are near-identical, (2) their distinctive name tokens overlap strongly, or
    (3) they share a developer AND a location token AND a distinctive name token."""
    if _norm(a["country"]) != _norm(b["country"]):
        return False
    name_sim = SequenceMatcher(None, _norm(a["project_name"]), _norm(b["project_name"])).ratio()
    na, nb = _tokens(a["project_name"]), _tokens(b["project_name"])
    if name_sim >= 0.85 or _jaccard(na, nb) >= 0.5:
        return True
    same_dev = bool(_tokens(a.get("developer")) & _tokens(b.get("developer")))
    loc_a = _tokens(a.get("city_or_region")) | na
    loc_b = _tokens(b.get("city_or_region")) | nb
    return same_dev and bool(na & nb) and bool(loc_a & loc_b)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df["project_id"] = None
    next_id = 1
    for i in range(len(df)):
        if df.at[i, "project_id"]:
            continue
        df.at[i, "project_id"] = f"DC{next_id:04d}"
        for j in range(i + 1, len(df)):
            if not df.at[j, "project_id"] and is_same_project(df.iloc[i], df.iloc[j]):
                df.at[j, "project_id"] = df.at[i, "project_id"]
        next_id += 1
    df["n_sources"] = df.groupby("project_id")["project_id"].transform("count")
    df = flag_conflicts(df)
    return df


def flag_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-source check: when sources disagree on capacity (>10%) or status for the same project, flag for review."""
    for pid, g in df.groupby("project_id"):
        if len(g) < 2:
            continue
        caps = g["capacity_mw"].dropna().astype(float)
        conflicts = []
        if len(caps) > 1 and (caps.max() - caps.min()) > 0.10 * caps.max():
            conflicts.append("capacity_conflict")
        if g["status"].nunique() > 1:
            conflicts.append("status_conflict")
        if conflicts:
            for idx in g.index:
                existing = df.at[idx, "qc_flags"]
                df.at[idx, "qc_flags"] = ";".join(filter(None, [existing] + conflicts))
                df.at[idx, "qc_status"] = "review"
    return df

# ---------------------------------------------------------------------------
# 5. EXPORT
# ---------------------------------------------------------------------------

def export(df: pd.DataFrame) -> None:
    out = df.drop(columns=["_source_text"], errors="ignore").copy()
    out["evidence"] = out["evidence"].apply(lambda e: json.dumps(e, ensure_ascii=False))
    cols = ["project_id"] + FIELDS + ["confidence", "qc_status", "qc_flags", "n_sources",
                                      "source_url", "source_file", "extracted_on", "notes", "evidence"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUT_DIR / "projects.csv", index=False)
    with pd.ExcelWriter(OUT_DIR / "projects.xlsx") as xw:
        out.to_excel(xw, sheet_name="projects", index=False)
        out[out.qc_status == "review"].to_excel(xw, sheet_name="needs_review", index=False)

    summary = {
        "records": len(out),
        "unique_projects": out["project_id"].nunique(),
        "qc_pass_rate": round((out.qc_status == "pass").mean(), 3) if len(out) else 0,
        "field_fill_rate": {f: round(out[f].notna().mean(), 3) for f in FIELDS if f in out},
        "top_flags": pd.Series(";".join(out.qc_flags).split(";")).replace("", pd.NA).dropna()
                       .str.split(":").str[0].value_counts().head(10).to_dict(),
    }
    (OUT_DIR / "qc_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

# ---------------------------------------------------------------------------
# 6. EVALUATE  (benchmark LLM output against human-labelled gold set)
# ---------------------------------------------------------------------------

def _match(pred, gold, field) -> bool:
    if pd.isna(gold) and pd.isna(pred):
        return True
    if pd.isna(gold) or pd.isna(pred):
        return False
    if field in ("capacity_mw", "investment_usd_m"):
        return abs(float(pred) - float(gold)) <= 0.05 * max(abs(float(gold)), 1)  # within 5%
    if field == "expected_online_year":
        return int(pred) == int(gold)
    if field == "status":
        return _norm(pred) == _norm(gold)
    # text fields: near-identical, or one side's distinctive tokens fully contained in the other's
    tp, tg = _tokens(pred), _tokens(gold)
    contained = bool(tp and tg) and (tg <= tp or tp <= tg)
    return contained or SequenceMatcher(None, _norm(pred), _norm(gold)).ratio() >= 0.8


def evaluate(gold_path: str) -> None:
    pred = pd.read_csv(OUT_DIR / "projects.csv")
    gold = pd.read_csv(gold_path)
    results = {f: [] for f in FIELDS}
    matched = 0
    for _, g in gold.iterrows():
        cands = pred[pred.source_file == g.source_file]
        if cands.empty:
            continue
        best = max(cands.itertuples(), key=lambda p: SequenceMatcher(
            None, _norm(p.project_name), _norm(g.project_name)).ratio())
        matched += 1
        for f in FIELDS:
            if f in gold.columns:
                results[f].append(_match(getattr(best, f), g[f], f))

    report = {
        "gold_records": len(gold),
        "recall_records_found": round(matched / len(gold), 3) if len(gold) else 0,
        "field_accuracy": {f: round(sum(v) / len(v), 3) for f, v in results.items() if v},
    }
    acc = [a for a in report["field_accuracy"].values()]
    report["overall_field_accuracy"] = round(sum(acc) / len(acc), 3) if acc else 0
    (OUT_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "extract", "evaluate"])
    ap.add_argument("--urls", default="data/urls.txt")
    ap.add_argument("--gold", default="data/gold/gold.csv")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached LLM responses and call the API again")
    a = ap.parse_args()
    {"collect": lambda: collect(a.urls),
     "extract": lambda: extract(use_cache=not a.no_cache),
     "evaluate": lambda: evaluate(a.gold)}[a.cmd]()
