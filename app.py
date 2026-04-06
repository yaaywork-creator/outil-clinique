import os
import re
import hmac
import uuid
import time
import math
import sqlite3
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

# =========================================================
# CONFIG
# =========================================================
load_dotenv()

APP_PASSWORD = os.getenv("APP_PASSWORD", "EDDAQAQ2026")
AZURE_DI_ENDPOINT = (os.getenv("AZURE_DI_ENDPOINT") or "").strip().rstrip("/")
AZURE_DI_KEY = (os.getenv("AZURE_DI_KEY") or "").strip()
AZURE_DI_API_VERSION = (os.getenv("AZURE_DI_API_VERSION") or "2024-11-30").strip()

GOOGLE_CSE_API_KEY = (os.getenv("GOOGLE_CSE_API_KEY") or "").strip()
GOOGLE_CSE_CX = (os.getenv("GOOGLE_CSE_CX") or "").strip()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_docs"
DB_FILE = DATA_DIR / "edd_exp_azure.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="EDDAQAQ EXP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #071120 0%, #0c1a33 100%);
        color: white;
    }
    .block-container {
        max-width: 1750px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px;
        min-height: 110px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .metric-title {
        font-size: 0.92rem;
        opacity: 0.8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.82rem;
        opacity: 0.75;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# AUTH
# =========================================================
def check_password():
    if st.session_state.get("authenticated", False):
        return True

    st.title("EDDAQAQ EXP")
    st.info("Accès sécurisé")

    pwd = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter", use_container_width=True):
        if hmac.compare_digest(pwd, APP_PASSWORD):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect")
    st.stop()

check_password()

# =========================================================
# HELPERS
# =========================================================
def azure_is_configured():
    return bool(AZURE_DI_ENDPOINT and AZURE_DI_KEY and AZURE_DI_API_VERSION)

def google_cse_is_configured():
    return bool(GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX)

def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def style_plot(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig

def normalize_text(value):
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value if value else None

def normalize_for_matching(s):
    if s is None or pd.isna(s):
        return ""
    s = str(s).lower().strip()
    replacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a",
        "î": "i", "ï": "i",
        "ô": "o",
        "ù": "u", "û": "u", "ü": "u",
        "ç": "c",
        "’": "'",
        "–": "-",
        "—": "-",
    }
    for a, b in replacements.items():
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def parse_numeric_value(x):
    if x is None or pd.isna(x):
        return np.nan

    s = str(x).strip().replace("\xa0", " ").replace(" ", "")
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s:
        return np.nan

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if s.count(",") == 1:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s and s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]

    try:
        return float(s)
    except Exception:
        return np.nan

def parse_date_fr(value):
    if value is None or pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    s = str(value).strip()
    if not s:
        return None

    s = s.replace(".", "/").replace("-", "/")
    s = re.sub(r"\s+", "", s)

    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt, errors="raise").date().isoformat()
        except Exception:
            pass

    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return dt.date().isoformat()
    except Exception:
        pass

    return None

def to_date_obj(x):
    d = parse_date_fr(x)
    if not d:
        return None
    return pd.to_datetime(d, errors="coerce")

def format_money(x):
    if x is None or pd.isna(x):
        return "-"
    try:
        return f"{float(x):,.2f}".replace(",", " ")
    except Exception:
        return str(x)

def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)

def save_uploaded_file(uploaded_file):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = UPLOAD_DIR / f"{stamp}_{safe_filename(uploaded_file.name)}"
    with open(out, "wb") as f:
        f.write(uploaded_file.getvalue())
    return str(out)

def infer_doc_type_from_name(name: str) -> str:
    n = normalize_for_matching(name)
    if "avoir" in n:
        return "facture_avoir"
    if "cotisation" in n:
        return "recu_cotisation"
    if "ordonnance" in n:
        return "ordonnance"
    if "bon de livraison" in n or "bl" in n:
        return "bon_livraison"
    return "facture"

def ceil_precise(x):
    try:
        return int(math.ceil(float(x)))
    except Exception:
        return 0

# =========================================================
# DB
# =========================================================
def get_db_connection():
    con = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=FULL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def safe_add_column(cur, table_name, column_name, column_type="TEXT"):
    cur.execute(f"PRAGMA table_info({table_name})")
    existing_cols = [row[1] for row in cur.fetchall()]
    if column_name not in existing_cols:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

def init_db():
    con = get_db_connection()
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents_achats (
            document_id TEXT PRIMARY KEY,
            source_file TEXT,
            stored_file_path TEXT,
            doc_type TEXT,
            supplier_name TEXT,
            issuer_name TEXT,
            invoice_number TEXT,
            document_date TEXT,
            due_date TEXT,
            client_name TEXT,
            payment_mode TEXT,
            total_ht REAL,
            total_tva REAL,
            total_ttc REAL,
            currency TEXT,
            raw_text TEXT,
            if_number TEXT,
            ice_number TEXT,
            rc_number TEXT,
            head_office_address TEXT,
            rc_city TEXT,
            created_at TEXT
        )
        """
    )

    for col in ["if_number", "ice_number", "rc_number", "head_office_address", "rc_city"]:
        safe_add_column(cur, "documents_achats", col, "TEXT")

    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_unique_invoice
        ON documents_achats (
            supplier_name,
            invoice_number,
            document_date,
            total_ttc
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS document_lines (
            line_id TEXT PRIMARY KEY,
            document_id TEXT,
            source_file TEXT,
            line_no INTEGER,
            reference TEXT,
            designation TEXT,
            service_date TEXT,
            quantity REAL,
            unit_price_ht REAL,
            unit_price_ttc REAL,
            line_amount_ht REAL,
            line_amount_ttc REAL,
            raw_line TEXT,
            FOREIGN KEY(document_id) REFERENCES documents_achats(document_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS supplier_payment_terms (
            term_id TEXT PRIMARY KEY,
            supplier_name TEXT,
            supplier_name_normalized TEXT,
            delay_days INTEGER,
            if_number TEXT,
            ice_number TEXT,
            rc_number TEXT,
            head_office_address TEXT,
            rc_city TEXT,
            notes TEXT,
            active INTEGER DEFAULT 1,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    for col in ["if_number", "ice_number", "rc_number", "head_office_address", "rc_city", "notes"]:
        safe_add_column(cur, "supplier_payment_terms", col, "TEXT")
    safe_add_column(cur, "supplier_payment_terms", "active", "INTEGER")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS payment_delay_records (
            record_id TEXT PRIMARY KEY,
            source_document_id TEXT,
            source_file TEXT,
            supplier_name TEXT,
            invoice_number TEXT,
            invoice_date TEXT,
            goods_nature TEXT,
            delivery_date TEXT,
            trans_month INTEGER,
            trans_year INTEGER,
            sector_payment_delay_days INTEGER,
            due_date TEXT,
            settlement_date TEXT,
            amount_ttc REAL,
            unpaid_amount REAL,
            late_months_unpaid INTEGER,
            amount_paid_late REAL,
            late_payment_date TEXT,
            payment_mode TEXT,
            payment_reference TEXT,
            pecuniary_fine_amount REAL,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS legal_params (
            param_key TEXT PRIMARY KEY,
            param_value TEXT,
            updated_at TEXT
        )
        """
    )

    con.commit()

    cur.execute("SELECT COUNT(*) FROM supplier_payment_terms")
    count_terms = cur.fetchone()[0]
    if count_terms == 0:
        supplier_terms = [
            ("MNC MEDICAL", 120, "", "", "", "", "", ""),
            ("MEDICA MAROC", 120, "", "", "", "", "", ""),
            ("PERFECT MED DISTRI", 120, "", "", "", "", "", ""),
            ("SAHARA REPARTITION DE MEDICAMENTS", 120, "", "", "", "", "", ""),
            ("MDK MEDICAL", 120, "", "", "", "", "", ""),
            ("TECMACO SERVICES", 120, "", "", "", "", "", ""),
            ("ANETT CASA", 90, "", "", "", "", "", ""),
            ("ANETT CASA", 120, "", "", "", "", "", "Uniquement sur l'exercice 2024 et sur les factures N°: 25050015 ; 25050009 ; 25050050"),
            ("COOPER PHARMA S.A", 90, "", "", "", "", "", ""),
            ("COSMO CHURGIE", 120, "", "", "", "", "", ""),
            ("MASTERLAB", 120, "", "", "", "", "", ""),
            ("BOTECH", 90, "", "", "", "", "", ""),
            ("EURO-BUREAU", 90, "", "", "", "", "", ""),
            ("POLYMEDIC", 90, "", "", "", "", "", ""),
            ("LAPROPHAN", 90, "", "", "", "", "", ""),
            ("BIOCROSS", 120, "", "", "", "", "", ""),
            ("HEMOLAB PHARMA", 90, "", "", "", "", "", ""),
            ("RADICMED", 120, "", "", "", "", "", ""),
            ("PROMAMEC", 120, "", "", "", "", "", ""),
            ("PROXIMAMEC", 120, "", "", "", "", "", ""),
            ("IMPRESSION ET CODE", 120, "", "", "", "", "", ""),
            ("REGENEXPERT", 90, "", "", "", "", "", ""),
            ("PROMEDIC", 120, "", "", "", "", "", ""),
            ("STERIPHARMA", 90, "", "", "", "", "", ""),
            ("ULTIMED", 120, "", "", "", "", "", ""),
            ("SODIMAREP", 90, "", "", "", "", "", ""),
            ("2J PRO", 120, "", "", "", "", "", ""),
            ("ORTHO SPOERT", 120, "", "", "", "", "", ""),
            ("FIRST CLASS MEDICAL", 120, "", "", "", "", "", ""),
            ("NAFISMED", 90, "", "", "", "", "", ""),
            ("NUMELEC MAROC", 120, "", "", "", "", "", ""),
            ("ORTHOMEDICAL SOLUTION", 120, "", "", "", "", "", ""),
            ("STE NOUVELLE PAPITERIE", 120, "", "", "", "", "", ""),
            ("SDIPH", 60, "", "", "", "", "", ""),
        ]
        now = datetime.now().isoformat(timespec="seconds")
        for supplier_name, delay_days, if_number, ice_number, rc_number, address, city, notes in supplier_terms:
            cur.execute(
                """
                INSERT INTO supplier_payment_terms (
                    term_id, supplier_name, supplier_name_normalized, delay_days,
                    if_number, ice_number, rc_number, head_office_address, rc_city,
                    notes, active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    supplier_name,
                    normalize_for_matching(supplier_name),
                    delay_days,
                    if_number,
                    ice_number,
                    rc_number,
                    address,
                    city,
                    notes,
                    1,
                    now,
                    now,
                ),
            )

    cur.execute("SELECT COUNT(*) FROM legal_params")
    count_params = cur.fetchone()[0]
    if count_params == 0:
        defaults = {
            "bam_rate_pct": "3.00",
            "default_delay_days": "60",
            "max_legal_delay_days": "120",
        }
        for k, v in defaults.items():
            cur.execute(
                "INSERT INTO legal_params (param_key, param_value, updated_at) VALUES (?, ?, ?)",
                (k, v, datetime.now().isoformat(timespec="seconds"))
            )

    con.commit()
    con.close()

init_db()

# =========================================================
# PARAMS
# =========================================================
def get_legal_params():
    con = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM legal_params", con)
    con.close()
    if df.empty:
        return {"bam_rate_pct": 3.0, "default_delay_days": 60, "max_legal_delay_days": 120}
    d = {r["param_key"]: r["param_value"] for _, r in df.iterrows()}
    return {
        "bam_rate_pct": parse_numeric_value(d.get("bam_rate_pct")) if d.get("bam_rate_pct") is not None else 3.0,
        "default_delay_days": int(parse_numeric_value(d.get("default_delay_days")) or 60),
        "max_legal_delay_days": int(parse_numeric_value(d.get("max_legal_delay_days")) or 120),
    }

def save_legal_params(bam_rate_pct, default_delay_days, max_legal_delay_days):
    con = get_db_connection()
    cur = con.cursor()
    params = {
        "bam_rate_pct": str(bam_rate_pct),
        "default_delay_days": str(int(default_delay_days)),
        "max_legal_delay_days": str(int(max_legal_delay_days)),
    }
    for k, v in params.items():
        cur.execute(
            """
            INSERT INTO legal_params (param_key, param_value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(param_key) DO UPDATE SET
                param_value = excluded.param_value,
                updated_at = excluded.updated_at
            """,
            (k, v, datetime.now().isoformat(timespec="seconds"))
        )
    con.commit()
    con.close()

# =========================================================
# AZURE ANALYSIS
# =========================================================
def guess_content_type_from_name(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".png":
        return "image/png"
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".tiff":
        return "image/tiff"
    if ext == ".bmp":
        return "image/bmp"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"

def azure_begin_invoice_analysis(file_bytes: bytes, filename: str) -> str:
    url = f"{AZURE_DI_ENDPOINT}/documentintelligence/documentModels/prebuilt-invoice:analyze?api-version={AZURE_DI_API_VERSION}"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_DI_KEY,
        "Content-Type": guess_content_type_from_name(filename),
    }
    resp = requests.post(url, headers=headers, data=file_bytes, timeout=120)
    resp.raise_for_status()
    operation_location = resp.headers.get("Operation-Location")
    if not operation_location:
        raise RuntimeError("Azure n'a pas renvoyé Operation-Location.")
    return operation_location

def azure_poll_result(operation_location: str, timeout_seconds: int = 180) -> dict:
    start = time.time()
    while True:
        resp = requests.get(
            operation_location,
            headers={"Ocp-Apim-Subscription-Key": AZURE_DI_KEY},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "").lower()

        if status == "succeeded":
            return data
        if status == "failed":
            raise RuntimeError(f"Azure a échoué : {data}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Délai dépassé pendant l'analyse Azure.")
        time.sleep(2)

def azure_analyze_invoice(file_bytes: bytes, filename: str) -> dict:
    operation_location = azure_begin_invoice_analysis(file_bytes, filename)
    return azure_poll_result(operation_location)

def get_field_obj(fields: dict, candidates: list):
    if not isinstance(fields, dict):
        return None
    normalized_map = {normalize_for_matching(k): v for k, v in fields.items()}
    for candidate in candidates:
        key = normalize_for_matching(candidate)
        if key in normalized_map:
            return normalized_map[key]
    return None

def field_to_text(field_obj):
    if not field_obj or not isinstance(field_obj, dict):
        return None
    if "content" in field_obj and field_obj["content"]:
        return str(field_obj["content"])
    if "valueString" in field_obj and field_obj["valueString"]:
        return str(field_obj["valueString"])
    if "valueDate" in field_obj and field_obj["valueDate"]:
        return str(field_obj["valueDate"])
    if "valueNumber" in field_obj and field_obj["valueNumber"] is not None:
        return str(field_obj["valueNumber"])
    if "valueCurrency" in field_obj and isinstance(field_obj["valueCurrency"], dict):
        amount = field_obj["valueCurrency"].get("amount")
        if amount is not None:
            return str(amount)
    return None

def field_to_number(field_obj):
    if not field_obj or not isinstance(field_obj, dict):
        return np.nan
    if "valueNumber" in field_obj and field_obj["valueNumber"] is not None:
        return parse_numeric_value(field_obj["valueNumber"])
    if "valueCurrency" in field_obj and isinstance(field_obj["valueCurrency"], dict):
        return parse_numeric_value(field_obj["valueCurrency"].get("amount"))
    if "content" in field_obj and field_obj["content"]:
        return parse_numeric_value(field_obj["content"])
    return np.nan

def field_currency_code(field_obj):
    if not field_obj or not isinstance(field_obj, dict):
        return None
    cur = field_obj.get("valueCurrency")
    if isinstance(cur, dict):
        return cur.get("currencyCode")
    return None

def field_to_date(field_obj):
    if not field_obj or not isinstance(field_obj, dict):
        return None
    if "valueDate" in field_obj and field_obj["valueDate"]:
        return str(field_obj["valueDate"])
    if "content" in field_obj and field_obj["content"]:
        return parse_date_fr(field_obj["content"])
    return None

def extract_raw_text_from_azure_result(result_json: dict) -> str:
    analyze_result = result_json.get("analyzeResult", {})
    content = analyze_result.get("content")
    if content:
        return content
    texts = []
    for page in analyze_result.get("pages", []):
        for line in page.get("lines", []):
            if line.get("content"):
                texts.append(line["content"])
    return "\n".join(texts)

def extract_regex_field(raw_text, patterns):
    if not raw_text:
        return None
    for p in patterns:
        m = re.search(p, raw_text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            val = normalize_text(m.group(1))
            if val:
                return val
    return None

def polygon_average_y(polygon):
    if not polygon or len(polygon) < 8:
        return None
    ys = []
    for i in range(1, len(polygon), 2):
        try:
            ys.append(float(polygon[i]))
        except Exception:
            pass
    if not ys:
        return None
    return sum(ys) / len(ys)

def extract_bottom_zone_text(result_json: dict, bottom_ratio: float = 0.72) -> str:
    analyze_result = result_json.get("analyzeResult", {})
    pages = analyze_result.get("pages", [])
    collected = []

    for page in pages:
        page_height = page.get("height")
        lines = page.get("lines", []) or []

        if not page_height:
            continue

        for line in lines:
            content = line.get("content", "")
            polygon = line.get("polygon") or []
            avg_y = polygon_average_y(polygon)

            if avg_y is not None and avg_y >= float(page_height) * bottom_ratio:
                if content:
                    collected.append(content)

    return "\n".join(collected)

def extract_identity_fields_from_footer(result_json: dict):
    footer_text = extract_bottom_zone_text(result_json, bottom_ratio=0.72)

    return {
        "if_number": extract_regex_field(footer_text, [
            r"\bN[°ºo]?\s*d[’' ]?IF\s*[:\-]?\s*([A-Za-z0-9\/\-.]+)",
            r"\bIF\s*[:\-]?\s*([A-Za-z0-9\/\-.]+)",
        ]),
        "ice_number": extract_regex_field(footer_text, [
            r"\bN[°ºo]?\s*d[’' ]?ICE\s*[:\-]?\s*([A-Za-z0-9\/\-.]+)",
            r"\bICE\s*[:\-]?\s*([A-Za-z0-9\/\-.]+)",
        ]),
        "rc_number": extract_regex_field(footer_text, [
            r"\bN[°ºo]?\s*RC\s*[:\-]?\s*([A-Za-z0-9\/\-.]+)",
            r"\bRC\s*[:\-]?\s*([A-Za-z0-9\/\-.]+)",
        ]),
        "head_office_address": extract_regex_field(footer_text, [
            r"Adresse\s+(?:du\s+)?si[eè]ge\s+social\s*[:\-]?\s*(.+)",
            r"Si[eè]ge\s+social\s*[:\-]?\s*(.+)",
            r"Si[eè]ge\s*social\s*[:\-]?\s*(.+)",
        ]),
        "rc_city": extract_regex_field(footer_text, [
            r"Ville\s+du\s+RC\s*[:\-]?\s*(.+)",
        ]),
    }

def normalize_azure_invoice_result(result_json: dict, filename: str, stored_path: str):
    analyze_result = result_json.get("analyzeResult", {})
    documents = analyze_result.get("documents", [])
    raw_text = extract_raw_text_from_azure_result(result_json)

    if not documents:
        raise RuntimeError("Azure n'a détecté aucun document facture.")

    doc0 = documents[0]
    fields = doc0.get("fields", {})

    vendor_name = field_to_text(get_field_obj(fields, ["VendorName", "SupplierName"]))
    customer_name = field_to_text(get_field_obj(fields, ["CustomerName", "Customer", "ClientName"]))
    invoice_id = field_to_text(get_field_obj(fields, ["InvoiceId", "InvoiceNumber"]))
    invoice_date = field_to_date(get_field_obj(fields, ["InvoiceDate"]))
    due_date = field_to_date(get_field_obj(fields, ["DueDate"]))
    sub_total = field_to_number(get_field_obj(fields, ["SubTotal"]))
    total_tax = field_to_number(get_field_obj(fields, ["TotalTax"]))
    amount_due = field_to_number(get_field_obj(fields, ["AmountDue", "InvoiceTotal", "Total"]))
    payment_terms = field_to_text(get_field_obj(fields, ["PaymentTerm", "PaymentTerms"]))
    currency = (
        field_currency_code(get_field_obj(fields, ["AmountDue", "InvoiceTotal", "Total"]))
        or field_currency_code(get_field_obj(fields, ["SubTotal"]))
        or "MAD"
    )

    ids = extract_identity_fields_from_footer(result_json)
    doc_type = infer_doc_type_from_name(filename)

    items_field = get_field_obj(fields, ["Items"])
    lines = []

    if items_field and isinstance(items_field, dict):
        array_items = items_field.get("valueArray") or []
        for idx, item in enumerate(array_items, start=1):
            item_obj = item.get("valueObject") or {}
            desc = field_to_text(get_field_obj(item_obj, ["Description", "ProductCode", "ItemDescription"]))
            qty = field_to_number(get_field_obj(item_obj, ["Quantity"]))
            unit_price = field_to_number(get_field_obj(item_obj, ["UnitPrice"]))
            amount = field_to_number(get_field_obj(item_obj, ["Amount", "TotalPrice"]))
            product_code = field_to_text(get_field_obj(item_obj, ["ProductCode"]))
            date_value = field_to_date(get_field_obj(item_obj, ["Date"]))

            lines.append({
                "line_no": idx,
                "reference": normalize_text(product_code),
                "designation": normalize_text(desc) or f"Ligne {idx}",
                "service_date": date_value,
                "quantity": qty,
                "unit_price_ht": unit_price,
                "unit_price_ttc": unit_price,
                "line_amount_ht": amount,
                "line_amount_ttc": amount,
                "raw_line": None,
            })

    if not lines:
        lines = [{
            "line_no": 1,
            "reference": None,
            "designation": normalize_text(vendor_name) or "Facture",
            "service_date": invoice_date,
            "quantity": 1.0,
            "unit_price_ht": amount_due if pd.notna(amount_due) else np.nan,
            "unit_price_ttc": amount_due if pd.notna(amount_due) else np.nan,
            "line_amount_ht": amount_due if pd.notna(amount_due) else np.nan,
            "line_amount_ttc": amount_due if pd.notna(amount_due) else np.nan,
            "raw_line": None,
        }]

    doc = {
        "source_file": filename,
        "stored_file_path": stored_path,
        "doc_type": doc_type,
        "supplier_name": normalize_text(vendor_name) or "Fournisseur non détecté",
        "issuer_name": normalize_text(vendor_name) or "Fournisseur non détecté",
        "invoice_number": normalize_text(invoice_id),
        "document_date": invoice_date,
        "due_date": due_date,
        "client_name": normalize_text(customer_name),
        "payment_mode": normalize_text(payment_terms),
        "total_ht": sub_total,
        "total_tva": total_tax,
        "total_ttc": amount_due,
        "currency": currency or "MAD",
        "raw_text": raw_text,
        "if_number": ids.get("if_number"),
        "ice_number": ids.get("ice_number"),
        "rc_number": ids.get("rc_number"),
        "head_office_address": ids.get("head_office_address"),
        "rc_city": ids.get("rc_city"),
    }

    return doc, lines

# =========================================================
# WEB ENRICHMENT
# =========================================================
def google_cse_search(query, num=5):
    if not google_cse_is_configured():
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": min(max(int(num), 1), 10),
        "hl": "fr",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("items", []) or []
    except Exception:
        return []

def extract_web_identity_from_text(text):
    if not text:
        return {}

    out = {}

    patterns = {
        "ice_number": [
            r"\bICE\s*[:\-]?\s*([0-9]{10,20})",
            r"\bIdentifiant\s+commun\s+de\s+l[’']entreprise\s*[:\-]?\s*([0-9]{10,20})",
        ],
        "if_number": [
            r"\bIF\s*[:\-]?\s*([0-9]{3,20})",
            r"\bIdentifiant\s+fiscal\s*[:\-]?\s*([0-9]{3,20})",
        ],
        "rc_number": [
            r"\bRC\s*[:\-]?\s*([0-9A-Za-z\/\-]{3,30})",
            r"\bRegistre\s+de\s+commerce\s*[:\-]?\s*([0-9A-Za-z\/\-]{3,30})",
        ],
    }

    for field, pats in patterns.items():
        for p in pats:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                out[field] = normalize_text(m.group(1))
                break

    addr = re.search(r"(adresse|si[eè]ge social)\s*[:\-]?\s*([^|•\n]{8,200})", text, flags=re.IGNORECASE)
    if addr:
        out["head_office_address"] = normalize_text(addr.group(2))

    return out

def enrich_supplier_row_from_web(row):
    supplier_name = normalize_text(row.get("supplier_name"))
    if not supplier_name:
        return row

    needs_any = any([
        not normalize_text(row.get("if_number")),
        not normalize_text(row.get("ice_number")),
        not normalize_text(row.get("rc_number")),
        not normalize_text(row.get("head_office_address")),
        not normalize_text(row.get("rc_city")),
    ])
    if not needs_any:
        return row

    queries = [
        f'"{supplier_name}" ICE IF RC adresse Maroc',
        f'"{supplier_name}" "Identifiant Fiscal" "ICE" "RC"',
        f'"{supplier_name}" "siège social" Maroc',
    ]

    found = {}
    for q in queries:
        results = google_cse_search(q, num=5)
        combined_text = ""
        for item in results:
            combined_text += " " + str(item.get("title", ""))
            combined_text += " " + str(item.get("snippet", ""))
            if "pagemap" in item:
                combined_text += " " + str(item.get("pagemap", ""))

        partial = extract_web_identity_from_text(combined_text)
        for k, v in partial.items():
            if v and not found.get(k):
                found[k] = v

    row["if_number"] = normalize_text(row.get("if_number")) or found.get("if_number")
    row["ice_number"] = normalize_text(row.get("ice_number")) or found.get("ice_number")
    row["rc_number"] = normalize_text(row.get("rc_number")) or found.get("rc_number")
    row["head_office_address"] = normalize_text(row.get("head_office_address")) or found.get("head_office_address")

    if not normalize_text(row.get("rc_city")) and normalize_text(row.get("head_office_address")):
        m = re.search(
            r"(casablanca|rabat|marrakech|fes|fès|agadir|tanger|meknes|mekn[eè]s|oujda|kenitra|k[eé]nitra|tetouan|tétouan|safi|el jadida|beni mellal|b[eé]ni mellal|nador|mohammedia)",
            row["head_office_address"],
            flags=re.IGNORECASE
        )
        if m:
            row["rc_city"] = normalize_text(m.group(1))

    return row

# =========================================================
# BUSINESS RULES
# =========================================================
def infer_supplier_convention(supplier_name, invoice_number=None, invoice_date=None):
    con = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM supplier_payment_terms WHERE active = 1", con)
    con.close()

    if df.empty:
        return None

    needed = [
        "supplier_name", "supplier_name_normalized", "delay_days",
        "if_number", "ice_number", "rc_number",
        "head_office_address", "rc_city", "notes"
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = None

    norm = normalize_for_matching(supplier_name)
    exact = df[df["supplier_name_normalized"] == norm].copy()

    if exact.empty:
        contains = df[df["supplier_name_normalized"].apply(lambda x: x in norm or norm in x if isinstance(x, str) else False)]
        exact = contains.copy()

    if exact.empty:
        return None

    if norm == normalize_for_matching("ANETT CASA") and invoice_number:
        inv = str(invoice_number)
        year = None
        if invoice_date:
            dt = to_date_obj(invoice_date)
            if dt is not None and pd.notna(dt):
                year = int(dt.year)

        special = exact[exact["notes"].fillna("").str.contains("2024", case=False, na=False)]
        if not special.empty and year == 2024 and any(x in inv for x in ["25050015", "25050009", "25050050"]):
            return special.iloc[0].to_dict()

        normal = exact[~exact["notes"].fillna("").str.contains("2024", case=False, na=False)]
        if not normal.empty:
            return normal.iloc[0].to_dict()

    return exact.iloc[0].to_dict()

def calculate_due_date(invoice_date, delay_days, default_delay_days=60, max_legal_delay_days=120):
    dt = to_date_obj(invoice_date)
    if dt is None or pd.isna(dt):
        return None

    if delay_days is None or pd.isna(delay_days):
        return None

    delay = parse_numeric_value(delay_days)
    if pd.isna(delay):
        return None

    if delay > max_legal_delay_days:
        delay = default_delay_days

    return (dt + pd.Timedelta(days=int(delay))).date().isoformat()

def compute_late_months_unpaid(unpaid_amount, due_date, settlement_date=None):
    unpaid_amount = parse_numeric_value(unpaid_amount)
    if pd.isna(unpaid_amount) or unpaid_amount <= 0:
        return 0

    due_dt = to_date_obj(due_date)
    ref_dt = to_date_obj(settlement_date) if settlement_date else pd.Timestamp(datetime.now().date())

    if due_dt is None or pd.isna(due_dt) or ref_dt is None or pd.isna(ref_dt):
        return 0

    days = (ref_dt - due_dt).days
    if days <= 0:
        return 0
    return ceil_precise(days / 30)

def build_goods_nature(lines_df, raw_text):
    if lines_df is not None and not lines_df.empty and "designation" in lines_df.columns:
        vals = lines_df["designation"].dropna().astype(str).tolist()
        vals = [v.strip() for v in vals if v.strip()]
        if vals:
            return " ; ".join(vals[:8])

    if raw_text:
        m = re.search(r"(objet|designation|nature)\s*[:\-]?\s*(.+)", raw_text, flags=re.IGNORECASE)
        if m:
            return normalize_text(m.group(2))
    return None

def upsert_supplier_in_conventions(doc):
    supplier_name = normalize_text(doc.get("supplier_name"))
    if not supplier_name:
        return

    con = get_db_connection()
    cur = con.cursor()
    norm = normalize_for_matching(supplier_name)

    cur.execute(
        "SELECT * FROM supplier_payment_terms WHERE supplier_name_normalized = ?",
        (norm,)
    )
    row = cur.fetchone()

    now = datetime.now().isoformat(timespec="seconds")

    if not row:
        cur.execute(
            """
            INSERT INTO supplier_payment_terms (
                term_id, supplier_name, supplier_name_normalized, delay_days,
                if_number, ice_number, rc_number, head_office_address, rc_city,
                notes, active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                supplier_name,
                norm,
                None,
                normalize_text(doc.get("if_number")),
                normalize_text(doc.get("ice_number")),
                normalize_text(doc.get("rc_number")),
                normalize_text(doc.get("head_office_address")),
                normalize_text(doc.get("rc_city")),
                None,
                1,
                now,
                now,
            )
        )
        con.commit()
    else:
        cur.execute(
            """
            UPDATE supplier_payment_terms
            SET
                if_number = COALESCE(NULLIF(if_number, ''), ?),
                ice_number = COALESCE(NULLIF(ice_number, ''), ?),
                rc_number = COALESCE(NULLIF(rc_number, ''), ?),
                head_office_address = COALESCE(NULLIF(head_office_address, ''), ?),
                rc_city = COALESCE(NULLIF(rc_city, ''), ?),
                updated_at = ?
            WHERE supplier_name_normalized = ?
            """,
            (
                normalize_text(doc.get("if_number")),
                normalize_text(doc.get("ice_number")),
                normalize_text(doc.get("rc_number")),
                normalize_text(doc.get("head_office_address")),
                normalize_text(doc.get("rc_city")),
                now,
                norm,
            )
        )
        con.commit()

    con.close()

def create_payment_delay_record_from_document(document_row, lines_df):
    params = get_legal_params()

    supplier_name = normalize_text(document_row.get("supplier_name"))
    invoice_number = normalize_text(document_row.get("invoice_number"))
    invoice_date = parse_date_fr(document_row.get("document_date"))
    raw_text = document_row.get("raw_text", "") or ""

    conv = infer_supplier_convention(supplier_name, invoice_number, invoice_date)
    delay_days = conv.get("delay_days") if conv else None

    goods_nature = build_goods_nature(lines_df, raw_text)

    delivery_date = invoice_date
    if lines_df is not None and not lines_df.empty and "service_date" in lines_df.columns:
        service_dates = lines_df["service_date"].dropna()
        if len(service_dates) > 0:
            delivery_date = pd.to_datetime(service_dates.iloc[0], errors="coerce")
            if pd.notna(delivery_date):
                delivery_date = delivery_date.date().isoformat()
            else:
                delivery_date = invoice_date

    due_date = calculate_due_date(
        invoice_date=invoice_date,
        delay_days=delay_days,
        default_delay_days=params["default_delay_days"],
        max_legal_delay_days=params["max_legal_delay_days"],
    )

    amount_ttc = parse_numeric_value(document_row.get("total_ttc"))
    unpaid_amount = amount_ttc if pd.notna(amount_ttc) else np.nan
    late_months_unpaid = compute_late_months_unpaid(unpaid_amount, due_date, None)

    return {
        "record_id": str(uuid.uuid4()),
        "source_document_id": document_row.get("document_id"),
        "source_file": document_row.get("source_file"),
        "supplier_name": supplier_name,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "goods_nature": goods_nature,
        "delivery_date": delivery_date,
        "trans_month": pd.to_datetime(invoice_date).month if invoice_date else None,
        "trans_year": pd.to_datetime(invoice_date).year if invoice_date else None,
        "sector_payment_delay_days": int(delay_days) if delay_days is not None and not pd.isna(delay_days) else None,
        "due_date": due_date,
        "settlement_date": None,
        "amount_ttc": amount_ttc,
        "unpaid_amount": amount_ttc if pd.notna(amount_ttc) else np.nan,
        "late_months_unpaid": late_months_unpaid,
        "amount_paid_late": 0.0,
        "late_payment_date": None,
        "payment_mode": normalize_text(document_row.get("payment_mode")),
        "payment_reference": None,
        "pecuniary_fine_amount": 0.0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

# =========================================================
# SAVE / LOAD
# =========================================================

def save_document_to_db(doc, lines_df):
    con = get_db_connection()
    cur = con.cursor()

    supplier_name = normalize_text(doc.get("supplier_name"))
    invoice_number = normalize_text(doc.get("invoice_number"))
    document_date = parse_date_fr(doc.get("document_date")) if doc.get("document_date") else None
    total_ttc = parse_numeric_value(doc.get("total_ttc"))

    cur.execute(
        """
        SELECT document_id
        FROM documents_achats
        WHERE COALESCE(supplier_name, '') = COALESCE(?, '')
          AND COALESCE(invoice_number, '') = COALESCE(?, '')
          AND COALESCE(document_date, '') = COALESCE(?, '')
          AND COALESCE(total_ttc, -999999999) = COALESCE(?, -999999999)
        LIMIT 1
        """,
        (supplier_name, invoice_number, document_date, total_ttc),
    )
    existing = cur.fetchone()

    now = datetime.now().isoformat(timespec="seconds")

    if existing:
        document_id = existing[0]

        cur.execute(
            """
            UPDATE documents_achats
            SET
                source_file = ?,
                stored_file_path = ?,
                doc_type = ?,
                supplier_name = ?,
                issuer_name = ?,
                invoice_number = ?,
                document_date = ?,
                due_date = ?,
                client_name = ?,
                payment_mode = ?,
                total_ht = ?,
                total_tva = ?,
                total_ttc = ?,
                currency = ?,
                raw_text = ?,
                if_number = ?,
                ice_number = ?,
                rc_number = ?,
                head_office_address = ?,
                rc_city = ?
            WHERE document_id = ?
            """,
            (
                doc.get("source_file"),
                doc.get("stored_file_path"),
                doc.get("doc_type"),
                supplier_name,
                doc.get("issuer_name"),
                invoice_number,
                document_date,
                parse_date_fr(doc.get("due_date")) if doc.get("due_date") else None,
                normalize_text(doc.get("client_name")),
                normalize_text(doc.get("payment_mode")),
                float(doc.get("total_ht")) if pd.notna(doc.get("total_ht")) else None,
                float(doc.get("total_tva")) if pd.notna(doc.get("total_tva")) else None,
                float(total_ttc) if pd.notna(total_ttc) else None,
                normalize_text(doc.get("currency")) or "MAD",
                doc.get("raw_text"),
                normalize_text(doc.get("if_number")),
                normalize_text(doc.get("ice_number")),
                normalize_text(doc.get("rc_number")),
                normalize_text(doc.get("head_office_address")),
                normalize_text(doc.get("rc_city")),
                document_id,
            ),
        )

        cur.execute("DELETE FROM document_lines WHERE document_id = ?", (document_id,))
    else:
        document_id = str(uuid.uuid4())

        cur.execute(
            """
            INSERT INTO documents_achats (
                document_id, source_file, stored_file_path, doc_type, supplier_name,
                issuer_name, invoice_number, document_date, due_date, client_name,
                payment_mode, total_ht, total_tva, total_ttc, currency, raw_text,
                if_number, ice_number, rc_number, head_office_address, rc_city, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                doc.get("source_file"),
                doc.get("stored_file_path"),
                doc.get("doc_type"),
                supplier_name,
                doc.get("issuer_name"),
                invoice_number,
                document_date,
                parse_date_fr(doc.get("due_date")) if doc.get("due_date") else None,
                normalize_text(doc.get("client_name")),
                normalize_text(doc.get("payment_mode")),
                float(doc.get("total_ht")) if pd.notna(doc.get("total_ht")) else None,
                float(doc.get("total_tva")) if pd.notna(doc.get("total_tva")) else None,
                float(total_ttc) if pd.notna(total_ttc) else None,
                normalize_text(doc.get("currency")) or "MAD",
                doc.get("raw_text"),
                normalize_text(doc.get("if_number")),
                normalize_text(doc.get("ice_number")),
                normalize_text(doc.get("rc_number")),
                normalize_text(doc.get("head_office_address")),
                normalize_text(doc.get("rc_city")),
                now,
            ),
        )

    if lines_df is not None and not lines_df.empty:
        temp_lines = lines_df.copy().reset_index(drop=True)
        for idx, row in temp_lines.iterrows():
            cur.execute(
                """
                INSERT INTO document_lines (
                    line_id, document_id, source_file, line_no, reference,
                    designation, service_date, quantity, unit_price_ht,
                    unit_price_ttc, line_amount_ht, line_amount_ttc, raw_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    document_id,
                    doc.get("source_file"),
                    int(row["line_no"]) if pd.notna(row.get("line_no")) else int(idx + 1),
                    normalize_text(row.get("reference")),
                    normalize_text(row.get("designation")),
                    parse_date_fr(row.get("service_date")) if row.get("service_date") else None,
                    float(row["quantity"]) if pd.notna(row.get("quantity")) else None,
                    float(row["unit_price_ht"]) if pd.notna(row.get("unit_price_ht")) else None,
                    float(row["unit_price_ttc"]) if pd.notna(row.get("unit_price_ttc")) else None,
                    float(row["line_amount_ht"]) if pd.notna(row.get("line_amount_ht")) else None,
                    float(row["line_amount_ttc"]) if pd.notna(row.get("line_amount_ttc")) else None,
                    normalize_text(row.get("raw_line")),
                ),
            )

    con.commit()
    con.close()

    upsert_supplier_in_conventions(doc)
    return document_id


def save_conventions_df(df):
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("DELETE FROM supplier_payment_terms")
    con.commit()

    now = datetime.now().isoformat(timespec="seconds")
    for _, row in df.iterrows():
        supplier_name = normalize_text(row.get("supplier_name"))
        if not supplier_name:
            continue

        delay_days = parse_numeric_value(row.get("delay_days"))
        active_val = parse_numeric_value(row.get("active"))

        cur.execute(
            """
            INSERT INTO supplier_payment_terms (
                term_id, supplier_name, supplier_name_normalized, delay_days,
                if_number, ice_number, rc_number, head_office_address, rc_city,
                notes, active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                supplier_name,
                normalize_for_matching(supplier_name),
                int(delay_days) if pd.notna(delay_days) else None,
                normalize_text(row.get("if_number")),
                normalize_text(row.get("ice_number")),
                normalize_text(row.get("rc_number")),
                normalize_text(row.get("head_office_address")),
                normalize_text(row.get("rc_city")),
                normalize_text(row.get("notes")),
                int(active_val) if pd.notna(active_val) else 1,
                now,
                now,
            )
        )
    con.commit()
    con.close()

def save_payment_delay_df(df):
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("DELETE FROM payment_delay_records")
    con.commit()

    now = datetime.now().isoformat(timespec="seconds")
    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO payment_delay_records (
                record_id, source_document_id, source_file, supplier_name,
                invoice_number, invoice_date, goods_nature, delivery_date,
                trans_month, trans_year, sector_payment_delay_days, due_date,
                settlement_date, amount_ttc, unpaid_amount, late_months_unpaid,
                amount_paid_late, late_payment_date, payment_mode, payment_reference,
                pecuniary_fine_amount, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalize_text(row.get("record_id")) or str(uuid.uuid4()),
                normalize_text(row.get("source_document_id")),
                normalize_text(row.get("source_file")),
                normalize_text(row.get("supplier_name")),
                normalize_text(row.get("invoice_number")),
                parse_date_fr(row.get("invoice_date")),
                normalize_text(row.get("goods_nature")),
                parse_date_fr(row.get("delivery_date")),
                int(parse_numeric_value(row.get("trans_month"))) if pd.notna(parse_numeric_value(row.get("trans_month"))) else None,
                int(parse_numeric_value(row.get("trans_year"))) if pd.notna(parse_numeric_value(row.get("trans_year"))) else None,
                int(parse_numeric_value(row.get("sector_payment_delay_days"))) if pd.notna(parse_numeric_value(row.get("sector_payment_delay_days"))) else None,
                parse_date_fr(row.get("due_date")),
                parse_date_fr(row.get("settlement_date")),
                parse_numeric_value(row.get("amount_ttc")),
                parse_numeric_value(row.get("unpaid_amount")),
                int(parse_numeric_value(row.get("late_months_unpaid"))) if pd.notna(parse_numeric_value(row.get("late_months_unpaid"))) else 0,
                parse_numeric_value(row.get("amount_paid_late")),
                parse_date_fr(row.get("late_payment_date")),
                normalize_text(row.get("payment_mode")),
                normalize_text(row.get("payment_reference")),
                parse_numeric_value(row.get("pecuniary_fine_amount")),
                now,
                now,
            )
        )
    con.commit()
    con.close()

def load_all_data():
    con = get_db_connection()
    docs = pd.read_sql_query("SELECT * FROM documents_achats", con)
    lines = pd.read_sql_query("SELECT * FROM document_lines", con)
    conventions = pd.read_sql_query("SELECT * FROM supplier_payment_terms ORDER BY supplier_name", con)
    payment_delay = pd.read_sql_query("SELECT * FROM payment_delay_records ORDER BY invoice_date DESC", con)
    con.close()

    if docs is not None and not docs.empty:
        for c in ["document_date", "due_date", "created_at"]:
            if c in docs.columns:
                docs[c] = pd.to_datetime(docs[c], errors="coerce")
        for c in ["total_ht", "total_tva", "total_ttc"]:
            if c in docs.columns:
                docs[c] = pd.to_numeric(docs[c], errors="coerce")

    if lines is not None and not lines.empty:
        for c in ["service_date"]:
            if c in lines.columns:
                lines[c] = pd.to_datetime(lines[c], errors="coerce")
        for c in ["quantity", "unit_price_ht", "unit_price_ttc", "line_amount_ht", "line_amount_ttc"]:
            if c in lines.columns:
                lines[c] = pd.to_numeric(lines[c], errors="coerce")

    if conventions is None or conventions.empty:
        conventions = pd.DataFrame(columns=[
            "supplier_name", "supplier_name_normalized", "delay_days",
            "if_number", "ice_number", "rc_number", "head_office_address",
            "rc_city", "notes", "active"
        ])
    else:
        needed_conv_cols = [
            "supplier_name", "supplier_name_normalized", "delay_days",
            "if_number", "ice_number", "rc_number", "head_office_address",
            "rc_city", "notes", "active"
        ]
        for col in needed_conv_cols:
            if col not in conventions.columns:
                conventions[col] = None

    if payment_delay is not None and not payment_delay.empty:
        date_cols = ["invoice_date", "delivery_date", "due_date", "settlement_date", "late_payment_date", "created_at", "updated_at"]
        for c in date_cols:
            if c in payment_delay.columns:
                payment_delay[c] = pd.to_datetime(payment_delay[c], errors="coerce")
        num_cols = [
            "sector_payment_delay_days", "amount_ttc", "unpaid_amount", "late_months_unpaid",
            "amount_paid_late", "pecuniary_fine_amount", "trans_month", "trans_year"
        ]
        for c in num_cols:
            if c in payment_delay.columns:
                payment_delay[c] = pd.to_numeric(payment_delay[c], errors="coerce")

    return docs, lines, conventions, payment_delay

# =========================================================
# BASE DOCUMENTS HELPERS
# =========================================================
def load_document_with_lines(document_id):
    con = get_db_connection()
    doc = pd.read_sql_query(
        "SELECT * FROM documents_achats WHERE document_id = ?",
        con,
        params=(document_id,)
    )
    lines = pd.read_sql_query(
        "SELECT * FROM document_lines WHERE document_id = ? ORDER BY line_no",
        con,
        params=(document_id,)
    )
    con.close()
    return doc, lines

def update_document_and_lines(document_id, doc_row, lines_df):
    con = get_db_connection()
    cur = con.cursor()

    cur.execute(
        """
        UPDATE documents_achats
        SET
            source_file = ?,
            doc_type = ?,
            supplier_name = ?,
            issuer_name = ?,
            invoice_number = ?,
            document_date = ?,
            due_date = ?,
            client_name = ?,
            payment_mode = ?,
            total_ht = ?,
            total_tva = ?,
            total_ttc = ?,
            currency = ?,
            if_number = ?,
            ice_number = ?,
            rc_number = ?,
            head_office_address = ?,
            rc_city = ?
        WHERE document_id = ?
        """,
        (
            normalize_text(doc_row.get("source_file")),
            normalize_text(doc_row.get("doc_type")),
            normalize_text(doc_row.get("supplier_name")),
            normalize_text(doc_row.get("supplier_name")),
            normalize_text(doc_row.get("invoice_number")),
            parse_date_fr(doc_row.get("document_date")),
            parse_date_fr(doc_row.get("due_date")),
            normalize_text(doc_row.get("client_name")),
            normalize_text(doc_row.get("payment_mode")),
            parse_numeric_value(doc_row.get("total_ht")),
            parse_numeric_value(doc_row.get("total_tva")),
            parse_numeric_value(doc_row.get("total_ttc")),
            normalize_text(doc_row.get("currency")) or "MAD",
            normalize_text(doc_row.get("if_number")),
            normalize_text(doc_row.get("ice_number")),
            normalize_text(doc_row.get("rc_number")),
            normalize_text(doc_row.get("head_office_address")),
            normalize_text(doc_row.get("rc_city")),
            document_id,
        )
    )

    cur.execute("DELETE FROM document_lines WHERE document_id = ?", (document_id,))

    if lines_df is not None and not lines_df.empty:
        temp_lines = lines_df.copy().reset_index(drop=True)
        for idx, row in temp_lines.iterrows():
            cur.execute(
                """
                INSERT INTO document_lines (
                    line_id, document_id, source_file, line_no, reference,
                    designation, service_date, quantity, unit_price_ht,
                    unit_price_ttc, line_amount_ht, line_amount_ttc, raw_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    document_id,
                    normalize_text(doc_row.get("source_file")),
                    int(idx + 1),
                    normalize_text(row.get("reference")),
                    normalize_text(row.get("designation")),
                    parse_date_fr(row.get("service_date")),
                    parse_numeric_value(row.get("quantity")),
                    parse_numeric_value(row.get("unit_price_ht")),
                    parse_numeric_value(row.get("unit_price_ttc")),
                    parse_numeric_value(row.get("line_amount_ht")),
                    parse_numeric_value(row.get("line_amount_ttc")),
                    normalize_text(row.get("raw_line")),
                )
            )

    con.commit()
    con.close()

    upsert_supplier_in_conventions({
        "supplier_name": normalize_text(doc_row.get("supplier_name")),
        "if_number": normalize_text(doc_row.get("if_number")),
        "ice_number": normalize_text(doc_row.get("ice_number")),
        "rc_number": normalize_text(doc_row.get("rc_number")),
        "head_office_address": normalize_text(doc_row.get("head_office_address")),
        "rc_city": normalize_text(doc_row.get("rc_city")),
    })

def delete_document_and_related_data(document_id):
    con = get_db_connection()
    cur = con.cursor()

    cur.execute("DELETE FROM payment_delay_records WHERE source_document_id = ?", (document_id,))
    cur.execute("DELETE FROM document_lines WHERE document_id = ?", (document_id,))
    cur.execute("DELETE FROM documents_achats WHERE document_id = ?", (document_id,))

    con.commit()
    con.close()

# =========================================================
# DELAI DISPLAY
# =========================================================
def build_payment_delay_display_df(payment_delay_df, docs_df, conventions_df):
    if payment_delay_df is None or payment_delay_df.empty:
        return pd.DataFrame()

    out = payment_delay_df.copy()
    docs = docs_df.copy() if docs_df is not None else pd.DataFrame()
    conv = conventions_df.copy() if conventions_df is not None else pd.DataFrame()

    for col in ["document_id", "if_number", "ice_number", "rc_number", "head_office_address", "rc_city"]:
        if col not in docs.columns:
            docs[col] = None

    out = out.merge(
        docs[["document_id", "if_number", "ice_number", "rc_number", "head_office_address", "rc_city"]],
        left_on="source_document_id",
        right_on="document_id",
        how="left",
        suffixes=("", "_doc")
    )

    for col in ["supplier_name", "supplier_name_normalized", "if_number", "ice_number", "rc_number", "head_office_address", "rc_city"]:
        if col not in conv.columns:
            conv[col] = None

    if not conv.empty:
        conv["supplier_name_normalized"] = conv["supplier_name"].apply(normalize_for_matching)
        conv = conv.drop_duplicates(subset=["supplier_name_normalized"], keep="first")
        out["supplier_name_normalized"] = out["supplier_name"].apply(normalize_for_matching)

        out = out.merge(
            conv[
                [
                    "supplier_name_normalized",
                    "if_number",
                    "ice_number",
                    "rc_number",
                    "head_office_address",
                    "rc_city",
                ]
            ],
            on="supplier_name_normalized",
            how="left",
            suffixes=("", "_conv")
        )

    for col in ["if_number", "ice_number", "rc_number", "head_office_address", "rc_city"]:
        if col not in out.columns:
            out[col] = None
        if f"{col}_conv" not in out.columns:
            out[f"{col}_conv"] = None
        out[col] = out[col].where(out[col].notna() & (out[col].astype(str).str.strip() != ""), out[f"{col}_conv"])

    final_df = pd.DataFrame({
        "N° d’IF": out["if_number"],
        "N° d'ICE": out["ice_number"],
        "Nom et prénom ou raison sociale": out["supplier_name"],
        "N° RC": out["rc_number"],
        "Adresse siège social": out["head_office_address"],
        "Ville du RC": out["rc_city"],
        "N° de facture": out["invoice_number"],
        "Date de facture": out["invoice_date"],
        "Nature des marchandises livrées, des travaux exécutés ou des services rendus": out["goods_nature"],
        "Date de livraison des marchandises, de l’exécution des travaux ou de la prestation de services": out["delivery_date"],
        "Mois (transactions d’une périodicité ne dépassant pas un mois)": out["trans_month"],
        "Année (transactions d’une périodicité ne dépassant pas un mois)2": out["trans_year"],
        "Délai de paiement des factures pour le secteur d’activité (2)": out["sector_payment_delay_days"],
        "Date d’Échéance": out["due_date"],
        "Date de Règlement": out["settlement_date"],
        "Montant de la facture TTC": out["amount_ttc"],
        "Montant non encore payé de la facture": out["unpaid_amount"],
        "Nombre des mois de retard afférent au montant non encore payé": out["late_months_unpaid"],
        "Montant payé totalement ou partiellement, hors délai au cours de la période objet de la déclaration": out["amount_paid_late"],
        "Date du paiement total ou partiel, hors délai": out["late_payment_date"],
        "Mode de paiement": out["payment_mode"],
        "Références du paiement": out["payment_reference"],
        "Montant de l’amende pécuniaire": out["pecuniary_fine_amount"],
        "_record_id": out["record_id"],
        "_source_document_id": out["source_document_id"],
        "_source_file": out["source_file"],
    })

    return final_df

def save_payment_delay_display_df(display_df):
    if display_df is None or display_df.empty:
        save_payment_delay_df(pd.DataFrame())
        return

    internal_df = pd.DataFrame({
        "record_id": display_df["_record_id"],
        "source_document_id": display_df["_source_document_id"],
        "source_file": display_df["_source_file"],
        "supplier_name": display_df["Nom et prénom ou raison sociale"],
        "invoice_number": display_df["N° de facture"],
        "invoice_date": display_df["Date de facture"],
        "goods_nature": display_df["Nature des marchandises livrées, des travaux exécutés ou des services rendus"],
        "delivery_date": display_df["Date de livraison des marchandises, de l’exécution des travaux ou de la prestation de services"],
        "trans_month": display_df["Mois (transactions d’une périodicité ne dépassant pas un mois)"],
        "trans_year": display_df["Année (transactions d’une périodicité ne dépassant pas un mois)2"],
        "sector_payment_delay_days": display_df["Délai de paiement des factures pour le secteur d’activité (2)"],
        "due_date": display_df["Date d’Échéance"],
        "settlement_date": display_df["Date de Règlement"],
        "amount_ttc": display_df["Montant de la facture TTC"],
        "unpaid_amount": display_df["Montant non encore payé de la facture"],
        "late_months_unpaid": display_df["Nombre des mois de retard afférent au montant non encore payé"],
        "amount_paid_late": display_df["Montant payé totalement ou partiellement, hors délai au cours de la période objet de la déclaration"],
        "late_payment_date": display_df["Date du paiement total ou partiel, hors délai"],
        "payment_mode": display_df["Mode de paiement"],
        "payment_reference": display_df["Références du paiement"],
        "pecuniary_fine_amount": display_df["Montant de l’amende pécuniaire"],
    })

    save_payment_delay_df(internal_df)

# =========================================================
# PENALITES
# =========================================================
def compute_penalties_df(payment_delay_df, bam_rate_pct):
    if payment_delay_df is None or payment_delay_df.empty:
        return pd.DataFrame(columns=[
            "Fournisseur", "N° de Facture", "Date de Facture", "Délais Convenus",
            "Date d’Échéance", "Date de Règlement", "Amende Applicable ?",
            "Taux BAM (%)", "Montant TTC", "Jours de Retard", "Mois Suppl.",
            "Taux de Pénalité (%)", "Pénalité", "Amende à Payer"
        ])

    df = payment_delay_df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
    df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
    df["amount_ttc"] = pd.to_numeric(df["amount_ttc"], errors="coerce")
    df["sector_payment_delay_days"] = pd.to_numeric(df["sector_payment_delay_days"], errors="coerce")

    today = pd.Timestamp(datetime.now().date())
    df["ref_payment_date"] = df["settlement_date"].fillna(today)

    df["Jours de Retard"] = np.where(
        (df["ref_payment_date"].notna()) & (df["due_date"].notna()),
        (df["ref_payment_date"] - df["due_date"]).dt.days,
        0
    )
    df["Jours de Retard"] = df["Jours de Retard"].clip(lower=0)

    df["Mois Suppl."] = np.where(
        df["Jours de Retard"] > 0,
        np.ceil(df["Jours de Retard"] / 30) - 1,
        0
    )
    df["Mois Suppl."] = df["Mois Suppl."].clip(lower=0)

    df["Taux BAM (%)"] = float(bam_rate_pct)
    df["Taux de Pénalité (%)"] = np.where(
        df["Jours de Retard"] > 0,
        df["Taux BAM (%)"] + 0.85 * df["Mois Suppl."],
        0
    )

    df["Pénalité"] = np.where(
        df["Jours de Retard"] > 0,
        (df["amount_ttc"].fillna(0) * df["Taux de Pénalité (%)"]) / 100,
        0
    )

    df["Amende à Payer"] = np.where(
        df["Mois Suppl."] > 0,
        (df["Jours de Retard"] * df["Pénalité"]) / 100,
        0
    )

    df["Amende Applicable ?"] = np.where(df["Jours de Retard"] > 0, "Oui", "Non")

    out = pd.DataFrame({
        "Fournisseur": df["supplier_name"],
        "N° de Facture": df["invoice_number"],
        "Date de Facture": df["invoice_date"].dt.date.astype(str),
        "Délais Convenus": df["sector_payment_delay_days"],
        "Date d’Échéance": df["due_date"].dt.date.astype(str),
        "Date de Règlement": df["settlement_date"].dt.date.astype(str),
        "Amende Applicable ?": df["Amende Applicable ?"],
        "Taux BAM (%)": df["Taux BAM (%)"],
        "Montant TTC": df["amount_ttc"],
        "Jours de Retard": df["Jours de Retard"],
        "Mois Suppl.": df["Mois Suppl."],
        "Taux de Pénalité (%)": df["Taux de Pénalité (%)"],
        "Pénalité": df["Pénalité"],
        "Amende à Payer": df["Amende à Payer"],
    })
    return out

# =========================================================
# REPORT
# =========================================================
def build_excel_report(docs_df, delay_export_df, penalties_df, conventions_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if docs_df is not None and not docs_df.empty:
            docs_df.to_excel(writer, sheet_name="Documents", index=False)
        if delay_export_df is not None and not delay_export_df.empty:
            delay_export_df.to_excel(writer, sheet_name="Delai_paiement", index=False)
        if penalties_df is not None and not penalties_df.empty:
            penalties_df.to_excel(writer, sheet_name="Penalites", index=False)
        if conventions_df is not None and not conventions_df.empty:
            conventions_df.to_excel(writer, sheet_name="Conventions", index=False)
    output.seek(0)
    return output.getvalue()

# =========================================================
# LOAD GLOBAL DATA
# =========================================================
docs_df, lines_df, conventions_df, payment_delay_df = load_all_data()
params = get_legal_params()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("EDDAQAQ EXP")
menu = st.sidebar.radio(
    "Navigation",
    [
        "Accueil",
        "Import & Extraction",
        "Base Documents",
        "Délai de paiement",
        "Pénalités",
        "Conventions délais fournisseurs",
    ],
)

if st.sidebar.button("Déconnexion", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

# =========================================================
# PAGE ACCUEIL
# =========================================================
if menu == "Accueil":
    st.title("Plateforme EDDAQAQ EXP")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Documents", str(len(docs_df)))
    with c2:
        metric_card("Montant total TTC", format_money(docs_df["total_ttc"].fillna(0).sum() if not docs_df.empty else 0))
    with c3:
        metric_card("Conventions", str(len(conventions_df)))
    with c4:
        metric_card("Lignes délai paiement", str(len(payment_delay_df)))

    st.markdown("### Paramètres légaux")
    p1, p2, p3 = st.columns(3)
    bam_rate_pct = p1.number_input("Taux BAM (%)", min_value=0.0, step=0.1, value=float(params["bam_rate_pct"]))
    default_delay_days = p2.number_input("Délai par défaut", min_value=0, step=1, value=int(params["default_delay_days"]))
    max_legal_delay_days = p3.number_input("Délai max légal", min_value=1, step=1, value=int(params["max_legal_delay_days"]))

    if st.button("Enregistrer les paramètres", use_container_width=True):
        save_legal_params(bam_rate_pct, default_delay_days, max_legal_delay_days)
        st.success("Paramètres enregistrés.")
        st.rerun()

# =========================================================
# PAGE IMPORT
# =========================================================
elif menu == "Import & Extraction":
    st.subheader("Import des documents")

    files = st.file_uploader(
        "Télécharger un ou plusieurs documents",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
    )

    if not azure_is_configured():
        st.warning("Azure n'est pas configuré. Remplis AZURE_DI_ENDPOINT et AZURE_DI_KEY dans le fichier .env pour l'extraction IA.")
        st.stop()

    if not files:
        st.info("Ajoute des fichiers pour lancer l’extraction.")
    else:
        for idx, file in enumerate(files, start=1):
            st.markdown(f"### Document {idx} - {file.name}")
            stored_path = save_uploaded_file(file)

            with st.spinner(f"Analyse Azure de {file.name}..."):
                try:
                    doc, lines = normalize_azure_invoice_result(
                        azure_analyze_invoice(file.getvalue(), file.name),
                        file.name,
                        stored_path
                    )
                except Exception as e:
                    st.error(f"Échec d'analyse pour {file.name} : {e}")
                    continue

            doc_editor = pd.DataFrame([{
                "source_file": doc.get("source_file"),
                "doc_type": doc.get("doc_type"),
                "supplier_name": doc.get("supplier_name"),
                "invoice_number": doc.get("invoice_number"),
                "document_date": doc.get("document_date"),
                "due_date": doc.get("due_date"),
                "client_name": doc.get("client_name"),
                "payment_mode": doc.get("payment_mode"),
                "total_ht": doc.get("total_ht"),
                "total_tva": doc.get("total_tva"),
                "total_ttc": doc.get("total_ttc"),
                "currency": doc.get("currency"),
                "if_number": doc.get("if_number"),
                "ice_number": doc.get("ice_number"),
                "rc_number": doc.get("rc_number"),
                "head_office_address": doc.get("head_office_address"),
                "rc_city": doc.get("rc_city"),
            }])

            lines_editor = pd.DataFrame(lines)

            with st.expander("Texte brut détecté"):
                st.text_area(
                    "Texte détecté par Azure",
                    doc.get("raw_text", ""),
                    height=300,
                    key=f"raw_text_{idx}",
                )

            st.markdown("#### En-tête document")
            edited_doc = st.data_editor(
                doc_editor,
                use_container_width=True,
                num_rows="fixed",
                key=f"doc_editor_{idx}",
            )

            st.markdown("#### Lignes détaillées")
            edited_lines = st.data_editor(
                lines_editor,
                use_container_width=True,
                num_rows="dynamic",
                key=f"lines_editor_{idx}",
            )

            row0 = edited_doc.iloc[0]

            doc_to_save = {
                "source_file": file.name,
                "stored_file_path": stored_path,
                "doc_type": normalize_text(row0.get("doc_type")),
                "supplier_name": normalize_text(row0.get("supplier_name")),
                "issuer_name": normalize_text(row0.get("supplier_name")),
                "invoice_number": normalize_text(row0.get("invoice_number")),
                "document_date": parse_date_fr(row0.get("document_date")) if row0.get("document_date") else None,
                "due_date": parse_date_fr(row0.get("due_date")) if row0.get("due_date") else None,
                "client_name": normalize_text(row0.get("client_name")),
                "payment_mode": normalize_text(row0.get("payment_mode")),
                "total_ht": parse_numeric_value(row0.get("total_ht")),
                "total_tva": parse_numeric_value(row0.get("total_tva")),
                "total_ttc": parse_numeric_value(row0.get("total_ttc")),
                "currency": normalize_text(row0.get("currency")) or "MAD",
                "raw_text": doc.get("raw_text", ""),
                "if_number": normalize_text(row0.get("if_number")),
                "ice_number": normalize_text(row0.get("ice_number")),
                "rc_number": normalize_text(row0.get("rc_number")),
                "head_office_address": normalize_text(row0.get("head_office_address")),
                "rc_city": normalize_text(row0.get("rc_city")),
            }

            if not edited_lines.empty:
                for c in ["quantity", "unit_price_ht", "unit_price_ttc", "line_amount_ht", "line_amount_ttc"]:
                    if c in edited_lines.columns:
                        edited_lines[c] = edited_lines[c].apply(parse_numeric_value)

            auto_save_key = f"auto_saved::{stored_path}"

            if st.button(f"Enregistrer / mettre à jour {file.name}", use_container_width=True, key=f"save_btn_{idx}"):
                doc_id = save_document_to_db(doc_to_save, edited_lines)
                st.session_state[auto_save_key] = True
                st.success(f"Document enregistré avec succès. ID : {doc_id}")
                st.rerun()

# =========================================================
# PAGE BASE DOCUMENTS
# =========================================================
elif menu == "Base Documents":
    st.subheader("Base documents")

    docs_df, lines_df, conventions_df, payment_delay_df = load_all_data()

    if docs_df.empty:
        st.info("Aucun document enregistré.")
        st.stop()

    show = docs_df.copy()
    for c in ["total_ht", "total_tva", "total_ttc"]:
        if c in show.columns:
            show[c] = show[c].apply(format_money)

    cols = [
        "document_id", "source_file", "supplier_name", "invoice_number",
        "document_date", "payment_mode", "total_ht", "total_tva", "total_ttc",
        "currency", "if_number", "ice_number", "rc_number",
        "head_office_address", "rc_city", "created_at"
    ]
    cols = [c for c in cols if c in show.columns]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)

    st.markdown("### Modifier un document")

    options_df = docs_df.copy()
    options_df["label"] = (
        options_df["supplier_name"].fillna("Sans fournisseur").astype(str)
        + " | "
        + options_df["invoice_number"].fillna("Sans numéro").astype(str)
        + " | "
        + options_df["source_file"].fillna("").astype(str)
    )

    selected_label = st.selectbox(
        "Choisir un document",
        options_df["label"].tolist()
    )

    selected_id = options_df.loc[options_df["label"] == selected_label, "document_id"].iloc[0]

    doc_one, lines_one = load_document_with_lines(selected_id)

    if doc_one.empty:
        st.warning("Document introuvable.")
        st.stop()

    doc_edit_cols = [
        "source_file", "doc_type", "supplier_name", "invoice_number",
        "document_date", "due_date", "client_name", "payment_mode",
        "total_ht", "total_tva", "total_ttc", "currency",
        "if_number", "ice_number", "rc_number",
        "head_office_address", "rc_city"
    ]

    for col in doc_edit_cols:
        if col not in doc_one.columns:
            doc_one[col] = None

    doc_editor = st.data_editor(
        doc_one[doc_edit_cols],
        use_container_width=True,
        num_rows="fixed",
        key="base_doc_editor"
    )

    line_edit_cols = [
        "reference", "designation", "service_date", "quantity",
        "unit_price_ht", "unit_price_ttc", "line_amount_ht",
        "line_amount_ttc", "raw_line"
    ]
    for col in line_edit_cols:
        if col not in lines_one.columns:
            lines_one[col] = None

    st.markdown("#### Lignes du document")
    lines_editor = st.data_editor(
        lines_one[line_edit_cols],
        use_container_width=True,
        num_rows="dynamic",
        key="base_lines_editor"
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Enregistrer les modifications du document", use_container_width=True):
            row0 = doc_editor.iloc[0].to_dict()
            update_document_and_lines(selected_id, row0, lines_editor)
            st.success("Document modifié avec succès.")
            st.rerun()

    with c2:
        confirm_delete = st.checkbox("Confirmer la suppression du document sélectionné")
        if st.button("Supprimer ce document", use_container_width=True, type="secondary"):
            if not confirm_delete:
                st.error("Coche d'abord la confirmation de suppression.")
            else:
                delete_document_and_related_data(selected_id)
                st.success("Document supprimé avec succès.")
                st.rerun()

# =========================================================
# PAGE DELAI DE PAIEMENT
# =========================================================
elif menu == "Délai de paiement":
    st.subheader("Rubrique délai de paiement")

    docs_df, lines_df, conventions_df, payment_delay_df = load_all_data()
    params = get_legal_params()

    if docs_df.empty:
        st.info("Aucun document importé.")
        st.stop()

    existing_ids = []
    if payment_delay_df is not None and not payment_delay_df.empty:
        existing_ids = payment_delay_df["source_document_id"].dropna().astype(str).tolist()

    missing_docs = docs_df[~docs_df["document_id"].astype(str).isin(existing_ids)].copy()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Créer les lignes délai de paiement manquantes", use_container_width=True):
            rows_to_add = []
            for _, drow in missing_docs.iterrows():
                doc_lines = lines_df[lines_df["document_id"] == drow["document_id"]].copy() if not lines_df.empty else pd.DataFrame()
                rows_to_add.append(create_payment_delay_record_from_document(drow, doc_lines))

            if rows_to_add:
                add_df = pd.DataFrame(rows_to_add)
                final_df = pd.concat([payment_delay_df, add_df], ignore_index=True) if payment_delay_df is not None and not payment_delay_df.empty else add_df
                save_payment_delay_df(final_df)
                st.success(f"{len(rows_to_add)} lignes créées.")
                st.rerun()
            else:
                st.info("Aucune nouvelle ligne à créer.")

    with c2:
        if st.button("Recharger", use_container_width=True):
            st.rerun()

    docs_df, lines_df, conventions_df, payment_delay_df = load_all_data()

    if payment_delay_df is None or payment_delay_df.empty:
        st.warning("Aucune ligne délai de paiement encore créée.")
        st.stop()

    display_df = build_payment_delay_display_df(payment_delay_df, docs_df, conventions_df)

    if display_df.empty:
        st.warning("Aucune donnée à afficher.")
        st.stop()

    for i in range(len(display_df)):
        row = display_df.iloc[i]
        due_date = calculate_due_date(
            invoice_date=row.get("Date de facture"),
            delay_days=row.get("Délai de paiement des factures pour le secteur d’activité (2)"),
            default_delay_days=params["default_delay_days"],
            max_legal_delay_days=params["max_legal_delay_days"],
        )
        display_df.at[i, "Date d’Échéance"] = due_date
        display_df.at[i, "Mois (transactions d’une périodicité ne dépassant pas un mois)"] = pd.to_datetime(parse_date_fr(row.get("Date de facture")), errors="coerce").month if parse_date_fr(row.get("Date de facture")) else None
        display_df.at[i, "Année (transactions d’une périodicité ne dépassant pas un mois)2"] = pd.to_datetime(parse_date_fr(row.get("Date de facture")), errors="coerce").year if parse_date_fr(row.get("Date de facture")) else None
        display_df.at[i, "Nombre des mois de retard afférent au montant non encore payé"] = compute_late_months_unpaid(
            row.get("Montant non encore payé de la facture"),
            due_date,
            row.get("Date de Règlement"),
        )

    visible_cols = [
        "N° d’IF",
        "N° d'ICE",
        "Nom et prénom ou raison sociale",
        "N° RC",
        "Adresse siège social",
        "Ville du RC",
        "N° de facture",
        "Date de facture",
        "Nature des marchandises livrées, des travaux exécutés ou des services rendus",
        "Date de livraison des marchandises, de l’exécution des travaux ou de la prestation de services",
        "Mois (transactions d’une périodicité ne dépassant pas un mois)",
        "Année (transactions d’une périodicité ne dépassant pas un mois)2",
        "Délai de paiement des factures pour le secteur d’activité (2)",
        "Date d’Échéance",
        "Date de Règlement",
        "Montant de la facture TTC",
        "Montant non encore payé de la facture",
        "Nombre des mois de retard afférent au montant non encore payé",
        "Montant payé totalement ou partiellement, hors délai au cours de la période objet de la déclaration",
        "Date du paiement total ou partiel, hors délai",
        "Mode de paiement",
        "Références du paiement",
        "Montant de l’amende pécuniaire",
        "_record_id",
        "_source_document_id",
        "_source_file",
    ]

    edited = st.data_editor(
        display_df[visible_cols],
        use_container_width=True,
        num_rows="dynamic",
        key="payment_delay_editor",
        column_config={
            "_record_id": None,
            "_source_document_id": None,
            "_source_file": None,
        }
    )

    if st.button("Enregistrer les modifications délai de paiement", use_container_width=True):
        for i in range(len(edited)):
            row = edited.iloc[i]
            due_date = calculate_due_date(
                invoice_date=row.get("Date de facture"),
                delay_days=row.get("Délai de paiement des factures pour le secteur d’activité (2)"),
                default_delay_days=params["default_delay_days"],
                max_legal_delay_days=params["max_legal_delay_days"],
            )
            edited.at[i, "Date d’Échéance"] = due_date
            edited.at[i, "Mois (transactions d’une périodicité ne dépassant pas un mois)"] = pd.to_datetime(parse_date_fr(row.get("Date de facture")), errors="coerce").month if parse_date_fr(row.get("Date de facture")) else None
            edited.at[i, "Année (transactions d’une périodicité ne dépassant pas un mois)2"] = pd.to_datetime(parse_date_fr(row.get("Date de facture")), errors="coerce").year if parse_date_fr(row.get("Date de facture")) else None
            edited.at[i, "Nombre des mois de retard afférent au montant non encore payé"] = compute_late_months_unpaid(
                row.get("Montant non encore payé de la facture"),
                due_date,
                row.get("Date de Règlement"),
            )

        save_payment_delay_display_df(edited)
        st.success("Table délai de paiement enregistrée.")
        st.rerun()

# =========================================================
# PAGE PENALITES
# =========================================================
elif menu == "Pénalités":
    st.subheader("Rubrique pénalités")

    docs_df, lines_df, conventions_df, payment_delay_df = load_all_data()
    params = get_legal_params()

    if payment_delay_df is None or payment_delay_df.empty:
        st.info("Aucune donnée dans la rubrique délai de paiement.")
        st.stop()

    bam_rate_pct = st.number_input(
        "Taux BAM (%) utilisé pour le calcul",
        min_value=0.0,
        step=0.1,
        value=float(params["bam_rate_pct"]),
        key="bam_penalty_rate"
    )

    penalties_df = compute_penalties_df(payment_delay_df, bam_rate_pct)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Montant TTC total", format_money(penalties_df["Montant TTC"].fillna(0).sum()))
    with k2:
        metric_card("Pénalités totales", format_money(penalties_df["Pénalité"].fillna(0).sum()))
    with k3:
        metric_card("Amendes totales", format_money(penalties_df["Amende à Payer"].fillna(0).sum()))
    with k4:
        metric_card("Factures hors délai", str(int((penalties_df["Jours de Retard"].fillna(0) > 0).sum())))

    st.dataframe(penalties_df, use_container_width=True, hide_index=True)

# =========================================================
# PAGE CONVENTIONS
# =========================================================
elif menu == "Conventions délais fournisseurs":
    st.subheader("Conventions fournisseurs")

    docs_df, lines_df, conventions_df, payment_delay_df = load_all_data()

    if conventions_df is None or conventions_df.empty:
        conventions_df = pd.DataFrame(columns=[
            "supplier_name", "delay_days", "if_number", "ice_number",
            "rc_number", "head_office_address", "rc_city", "notes", "active"
        ])

    st.info("Tu peux modifier ici manuellement les données. Si la facture ne contient pas IF / ICE / RC / adresse / ville, l'app les reprend ici.")

    conv_cols = [
        "supplier_name",
        "delay_days",
        "if_number",
        "ice_number",
        "rc_number",
        "head_office_address",
        "rc_city",
        "notes",
        "active",
    ]
    for col in conv_cols:
        if col not in conventions_df.columns:
            conventions_df[col] = None

    edited_terms = st.data_editor(
        conventions_df[conv_cols],
        use_container_width=True,
        num_rows="dynamic",
        key="terms_editor"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Enregistrer les conventions fournisseurs", use_container_width=True):
            save_conventions_df(edited_terms)
            st.success("Conventions enregistrées.")
            st.rerun()

    with col_b:
        if st.button("Compléter automatiquement les conventions via web", use_container_width=True):
            if not google_cse_is_configured():
                st.error("Tu dois remplir GOOGLE_CSE_API_KEY et GOOGLE_CSE_CX dans le fichier .env")
            else:
                enriched = edited_terms.copy()
                progress = st.progress(0)
                total = len(enriched) if len(enriched) > 0 else 1

                for i in range(len(enriched)):
                    row = enriched.iloc[i].copy()
                    row = enrich_supplier_row_from_web(row)
                    for col in enriched.columns:
                        enriched.at[i, col] = row.get(col)
                    progress.progress((i + 1) / total)

                save_conventions_df(enriched)
                st.success("Enrichissement web terminé. Vérifie quand même les résultats.")
                st.rerun()

    delay_export_df = build_payment_delay_display_df(payment_delay_df, docs_df, conventions_df)
    delay_export_df = delay_export_df.drop(columns=["_record_id", "_source_document_id", "_source_file"], errors="ignore")

    penalties_df = compute_penalties_df(payment_delay_df, get_legal_params()["bam_rate_pct"]) if payment_delay_df is not None and not payment_delay_df.empty else pd.DataFrame()
    excel_bytes = build_excel_report(docs_df, delay_export_df, penalties_df, edited_terms)

    st.download_button(
        "Télécharger l’export Excel global",
        data=excel_bytes,
        file_name="rapport_global_delai_penalites.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )