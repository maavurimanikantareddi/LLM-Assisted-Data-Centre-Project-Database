# Data Centre Project Extractor

An LLM-assisted pipeline that turns unstructured public sources (news articles, company announcements, planning documents) into a **structured, validated and auditable project-level dataset of data centre assets**.

## Why
AI and cloud growth is driving rapid data centre build-out. Information about individual projects (capacity, developer, status, power supply) is scattered across unstructured text. This project tests how reliably a large language model can extract that information, and how to build quality controls so the output can be trusted.

## Pipeline

```
collect  ->  extract (LLM)  ->  validate (QC)  ->  deduplicate  ->  evaluate  ->  export
```

| Stage | What it does |
|---|---|
| **Collect** | Fetches web pages with `requests` + `BeautifulSoup`, strips boilerplate, stores clean text with its source URL |
| **Extract** | Prompt-engineered LLM call (temperature 0, strict JSON schema) pulls out project name, developer, operator, location, capacity (MW), status, online year, power supply and investment, plus a **verbatim evidence quote for every field** |
| **Validate** | Rule-based QC: required fields, allowed status values, capacity and year range checks, **hallucination check** (every evidence quote must appear in the source text), unsourced-field detection, low-confidence flag |
| **Deduplicate** | Fuzzy matching on project name + country merges the same project reported by several sources, assigns a stable `project_id` and counts sources |
| **Evaluate** | Benchmarks LLM output against a hand-labelled gold set: record recall and field-level accuracy |
| **Export** | `projects.csv`, `projects.xlsx` (with a `needs_review` sheet for human-in-the-loop review), `qc_report.json`, `eval_report.json` |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key      # Windows: set ANTHROPIC_API_KEY=your_key
```

The LLM call sits in one function (`call_llm`), so it's easy to swap in OpenAI or Gemini.

## Run

```bash
# 1. Put source URLs in data/urls.txt (one per line), then:
python dc_extractor.py collect

#    ...or paste article text into data/raw/*.txt (first line: SOURCE_URL: <url>)

# 2. Extract, validate, deduplicate, export
python dc_extractor.py extract

# 3. Label 20-30 records by hand in data/gold/gold.csv, then benchmark
python dc_extractor.py evaluate
```

`gold.csv` columns: `source_file, project_name, country, capacity_mw, status, expected_online_year` (add any other fields you labelled).

## Results (run of November 2026)

**Data:** 36 public news and press sources (Data Centre Review, DatacenterDynamics, The Register, Reuters, Baxtel, Data Center Frontier and others) covering UK, European and US data centre projects announced in 2025–26.

| Metric | Result |
|---|---|
| Sources processed | 36 |
| Records extracted | 43 |
| Unique projects after deduplication | 29 (10 reported by 2+ sources) |
| Countries covered | 8 (UK 12, US 10, Finland 2, Netherlands, Germany, France, Poland, Switzerland) |
| Capacity tracked | ~10.5 GW across 24 projects with a stated capacity |
| Evidence quotes checked against source text | 294 (0 not found, so no hallucinated quotes) |
| Records passing all QC checks | 25 of 43 (58%); 18 routed to human review |
| Field-level accuracy vs 30-record labelled sample | 99% overall (capacity 93%, all other fields 100%) |
| Record recall vs labelled sample | 100% |

**What the QC caught**
- **Cross-source conflicts:** 8 records with capacity disagreements between sources (e.g. Pure DC Amsterdam reported as 78 MW by two sources and 70 MW by a third; DC01UK as 250 MW vs an older 320 MW target) and 12 with status disagreements as projects moved through planning.
- **Unsourced fields:** 9 values the model filled in without a supporting quote (mostly country inferred from a city name).
- **Low-confidence record:** 1 (an unnamed Virtus site), which was also wrongly merged with Virtus Wustermark by deduplication. The capacity-conflict check caught this.

**Accuracy errors (2):** both on capacity, where the source gave an outdated target (DC01UK 320 MW) or a power-delivery milestone (Fermi America "1 GW of power by end of next year") rather than current campus capacity. The model had already lowered its confidence and written notes on both.

**Iteration:** the first deduplication pass used only name similarity and found 36 "unique" projects (it missed matches like "Frontier" vs "Frontier campus"). Switching to distinctive-token matching plus developer and location overlap brought this to 29, and a cross-source conflict check was added to catch false merges.

**Method note:** LLM responses are cached in `output/llm_cache/`, so the run is reproducible without API cost. Run `python dc_extractor.py extract --no-cache` to re-extract with a live model.

## Limitations / next steps
- Add planning portals and utility interconnection queues as sources
- Field-specific prompts for harder fields (status, power supply)
- Cross-check capacity against a second model or source
