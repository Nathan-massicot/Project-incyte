# Projet Incyte – Data & Model Scripts
"""
This single Python module groups together three command-line utilities that cover
all deliverables for the « Données & modèle » work-package:

1. **extract_ema_requirements** ― builds *requirements.csv* by scraping official
   EMA / EC sources (EudraLex Vol 2B, EU Module 1 eCTD spec v3.1, validation
   criteria, etc.).
2. **init_db** ― creates a fresh SQLite database (project.db) with the required
   schema: requirement, submission, task, user, kpi.
3. **seed_example** ― registers the fictitious orphan submission “Mol_Alpha” and
   populates the database with the first 15 tasks for illustration.

Usage
-----
```bash
python project_data_model.py extract_ema_requirements  # → data/requirements.csv
python project_data_model.py init_db                   # → project.db
python project_data_model.py seed_example              # uses requirements.csv & project.db
```
Each sub-command can be run independently from the others.
"""

import argparse
import csv
import datetime as dt
import io
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

###############################################################################
# Configuration constants
###############################################################################
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = Path("project.db")
REQUIREMENTS_CSV = DATA_DIR / "requirements.csv"

# Official reference URLs (centralised – human – orphan focus)
EMA_SOURCES = [
    "https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-2_en",
    "https://esubmission.ema.europa.eu/eumodule1/EU%20M1%20eCTD%20Spec%20v3.1%20-%20June%202024%20-%20final%20version.pdf",
    # "https://esubmission.ema.europa.eu/eumodule1/docs/eCTD%20EU%20Validation%20Criteria%20v8.2%20-%20June%202025.xlsx",  # temporairement désactivé
]

###############################################################################
# Helper functions – extraction layer
###############################################################################

def _download(url: str) -> bytes:
    """Download a remote resource and return the raw content."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _parse_html_tables(html: str) -> Iterable[pd.DataFrame]:
    """Return all <table> elements from an HTML page as DataFrames."""
    soup = BeautifulSoup(html, "lxml")
    for table in soup.find_all("table"):
        yield pd.read_html(str(table), flavor="bs4")[0]


def _parse_pdf_tables(pdf_bytes: bytes) -> Iterable[pd.DataFrame]:
    """Extract tables from PDFs using tabula-py if available, otherwise camelot."""
    import tempfile
    import os
    import pandas as pd

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        try:
            from tabula.io import read_pdf as tabula_read_pdf
            dfs = tabula_read_pdf(tmp_path, pages="all", multiple_tables=True)
            for df in dfs:
                if isinstance(df, pd.DataFrame):
                    yield df
        except Exception:
            try:
                from camelot.io import read_pdf as camelot_read_pdf
                tables = camelot_read_pdf(tmp_path, pages="all")
                for t in tables:
                    if hasattr(t, 'df') and isinstance(t.df, pd.DataFrame):
                        yield t.df
            except Exception:
                pass
    finally:
        os.unlink(tmp_path)

###############################################################################
# 1️⃣  extract_ema_requirements
###############################################################################

def extract_ema_requirements(out: Path = REQUIREMENTS_CSV) -> None:
    """Scrape the official EMA sources and build the consolidated CSV."""

    frames: List[pd.DataFrame] = []

    for url in EMA_SOURCES:
        print(f"[extract] Fetching {url}…", file=sys.stderr)
        content = _download(url)

        if url.lower().endswith(('.pdf', '.PDF')):
            dfs = list(_parse_pdf_tables(content))
        elif url.lower().endswith(('.xls', '.xlsx')):
            dfs = [pd.read_excel(io.BytesIO(content))]
        else:
            dfs = list(_parse_html_tables(content.decode("utf-8", "ignore")))

        print(f"[DEBUG] {url} : {len(dfs)} tableaux extraits")
        if dfs:
            print(f"[DEBUG] Colonnes du premier tableau: {dfs[0].columns}")
        frames.extend(dfs)

    if not frames:
        print("[DEBUG] Aucun tableau extrait de toutes les sources !")
        raise RuntimeError("No tables found in the provided sources — please verify URLs.")

    raw = pd.concat(frames, ignore_index=True)
    print(f"[DEBUG] Nombre de lignes concaténées: {len(raw)}")
    print(raw.head())

    # Normalisation heuristics – keep only CTD-like rows and clean headers
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    # Expected columns to map / guess
    mapping_cols = {
        "module": "module",
        "section": "section",
        "document": "document_name",
        "document_name": "document_name",
        "mandatory": "mandatory",
    }

    df = pd.DataFrame(columns=pd.Index([
        "module", "section", "document_name", "mandatory",
        "template_available", "language_specific", "orphan_specific",
        "default_leadtime_days", "criticality"
    ]))

    for _, row in raw.iterrows():
        # Heuristic: recognise CTD section numbers (e.g. "1.3.1" or "Module 2.5")
        section_match = re.search(r"(\d+(?:\.\d+)+)", " ".join(map(str, row.values)))
        module = section_match.group(0).split(".")[0] if section_match else ""

        doc_name = row.get("document") or row.get("document_name") or ""
        mandatory_flag = bool(re.search(r"(shall|must|mandatory|required)", str(row), re.I))

        # Valeur par défaut
        leadtime = 45
        if "default_leadtime_days" in row:
            try:
                leadtime = int(row["default_leadtime_days"])
            except Exception:
                pass

        # Calcul du score de criticité
        if mandatory_flag and leadtime <= 30:
            criticality = 3
        elif mandatory_flag and leadtime > 30:
            criticality = 2
        else:
            criticality = 1

        df.loc[len(df)] = {
            "module": module,
            "section": section_match.group(0) if section_match else "",
            "document_name": doc_name.strip(),
            "mandatory": "Y" if mandatory_flag else "N",
            "template_available": "",  # could be filled manually or via another scraper
            "language_specific": "N",
            "orphan_specific": "Y" if re.search(r"orphan", str(row), re.I) else "N",
            "default_leadtime_days": leadtime,  # default assumption; refine later
            "criticality": criticality,
        }

    # Drop empty document rows and duplicates
    df = df[df.document_name != ""].drop_duplicates().sort_values(by=["module", "section"])  # type: ignore[arg-type]

    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✔ requirements.csv written to {out}")

###############################################################################
# 2️⃣  init_db – database schema (SQLite)
###############################################################################

SCHEMA_SQL = r"""
-- Table: requirement
CREATE TABLE IF NOT EXISTS requirement (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    module          TEXT NOT NULL,
    section         TEXT NOT NULL,
    document_name   TEXT NOT NULL,
    mandatory       TEXT CHECK(mandatory IN ('Y', 'N')) NOT NULL,
    template_url    TEXT,
    language_specific TEXT CHECK(language_specific IN ('Y', 'N')) DEFAULT 'N',
    orphan_specific TEXT CHECK(orphan_specific IN ('Y', 'N')) DEFAULT 'N',
    default_leadtime_days INTEGER DEFAULT 45
);

-- Table: submission
CREATE TABLE IF NOT EXISTS submission (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    molecule_name    TEXT NOT NULL,
    procedure_type   TEXT NOT NULL,
    orphan           TEXT CHECK(orphan IN ('Y', 'N')) DEFAULT 'N',
    ema_number       TEXT,
    start_date       DATE,
    validation_date  DATE,
    clock_stop_date  DATE,
    status           TEXT DEFAULT 'In preparation'
);

-- Table: task
CREATE TABLE IF NOT EXISTS task (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id   INTEGER NOT NULL REFERENCES submission(id) ON DELETE CASCADE,
    requirement_id  INTEGER REFERENCES requirement(id),
    name            TEXT NOT NULL,
    due_date        DATE,
    status          TEXT DEFAULT 'Not started',
    assigned_user_id INTEGER REFERENCES user(id)
);

-- Table: user
CREATE TABLE IF NOT EXISTS user (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    role    TEXT,
    email   TEXT UNIQUE
);

-- Table: kpi
CREATE TABLE IF NOT EXISTS kpi (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id   INTEGER NOT NULL REFERENCES submission(id) ON DELETE CASCADE,
    metric_name     TEXT NOT NULL,
    value           REAL NOT NULL,
    calculated_at   DATE DEFAULT CURRENT_DATE
);
"""

def init_db(db_path: Path = DB_PATH):
    """Create the SQLite schema as per SCHEMA_SQL."""
    print(f"[init_db] Creating database at {db_path}…", file=sys.stderr)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
    print("✔ Database initialised.")

###############################################################################
# 3️⃣  seed_example – Mol_Alpha orphan submission
###############################################################################

FIRST_TASKS = [
    "Prepare application form (Module 1)",
    "Draft SmPC (Module 1.3)",
    "Draft RMP (Module 1.8.1)",
    "Assemble Quality Overall Summary (Module 2.3)",
    "Assemble Non-clinical Overview (Module 2.4)",
    "Assemble Clinical Overview (Module 2.5)",
    "Compile Module 3 – S Drug Substance", 
    "Compile Module 3 – P Drug Product", 
    "Generate CSR for pivotal study", 
    "Translate Product Information", 
    "User-patient testing of PL", 
    "Upload literature references", 
    "Hyperlink Module 2 documents", 
    "Internal QC of eCTD build", 
    "Pre-submission meeting with EMA",
]

def seed_example(requirements_csv: Path = REQUIREMENTS_CSV, db_path: Path = DB_PATH):
    """Seed the DB with a fictitious orphan submission and starter tasks."""

    if not db_path.exists():
        raise FileNotFoundError("Database not found. Run init_db first.")

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # 1. Insert submission Mol_Alpha
        cur.execute(
            """
            INSERT INTO submission (molecule_name, procedure_type, orphan, start_date, status)
            VALUES (?, ?, 'Y', ?, 'Draft')
            """,
            ("Mol_Alpha", "Centralised", dt.date.today().isoformat()),
        )
        submission_id = cur.lastrowid

        # 2. Map first 15 requirements to tasks (loose link for demo)
        cur.execute("SELECT id, document_name FROM requirement LIMIT 15")
        reqs = cur.fetchall()

        for i, task_name in enumerate(FIRST_TASKS):
            req_id = reqs[i][0] if i < len(reqs) else None
            due = dt.date.today() + dt.timedelta(days=45)
            cur.execute(
                """
                INSERT INTO task (submission_id, requirement_id, name, due_date)
                VALUES (?, ?, ?, ?)
                """,
                (submission_id, req_id, task_name, due.isoformat()),
            )

        conn.commit()

    print("✔ Mol_Alpha seeded with 15 tasks.")

###############################################################################
# Entrypoint
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="EMA submission data-model utilities")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("extract_ema_requirements")
    subparsers.add_parser("init_db")
    subparsers.add_parser("seed_example")

    args = parser.parse_args()

    if args.cmd == "extract_ema_requirements":
        extract_ema_requirements()
    elif args.cmd == "init_db":
        init_db()
    elif args.cmd == "seed_example":
        seed_example()

if __name__ == "__main__":
    main()
