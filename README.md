# ICU Agent OS

A single-app clinical decision-support tool for the ICU. Built so that facts are
never left to an LLM: severity scores are computed deterministically, guidelines
are retrieved from a local store, and the LLM (optional) only *phrases* text.

## What it does
- **Deterministic scores** — qSOFA, full SOFA, real APACHE II (verified against
  hand calculations; missing inputs are flagged, impossible inputs raise).
- **Guideline lookup** — keyword search over a local, editable store. The LLM can
  only summarise text that is physically present, so it cannot invent a protocol.
- **Differential support** — rule-based ranking from selected findings, optionally
  refined by the LLM. Support only, never a final diagnosis.
- **Progress notes** — SOAP note assembled from structured data; LLM polish optional.
- **Persistent memory** — stored in its OWN table, never overloaded into the
  clinical note field.

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```
Windows: double-click `run_icu_agent_os.bat`.

## Data protection (DPDP Act)
Every prompt is PHI-redacted before leaving the process (`phi.py`). For the
strongest posture, point the LLM at a **local model** so no patient data ever
leaves the hospital: in the sidebar set Base URL to `http://localhost:11434/v1`
(Ollama) — no API key needed. With no LLM configured, the whole app still works;
only text drafting falls back to templates.

## Tests
```
python run_tests.py
```
Covers scores, PHI redaction, database + memory, guideline search, router, and
the offline LLM fallback — all with no network and no API key.

## Files (one of each — no version sprawl)
| file | role |
|------|------|
| `app.py` | the only Streamlit entry point |
| `icu_scores.py` | deterministic scoring engine |
| `guidelines.py` | local guideline store + search |
| `agents.py` | router + specialist agents |
| `database.py` | SQLite schema + CRUD |
| `phi.py` | identifier redaction |
| `llm.py` | provider-agnostic, offline-safe LLM client |

## Important
Guideline text and score interpretations are compact summaries and must be
locally validated against your unit protocols and current editions before
clinical use. This tool supports decisions; it does not make them.
