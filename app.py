import os
import re
import hmac
import uuid
import time
import sqlite3
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# =========================================================
# ENV / CONFIG
# =========================================================
load_dotenv()

APP_PASSWORD = os.getenv("APP_PASSWORD", "EDDAQAQ2026")
AZURE_DI_ENDPOINT = (os.getenv("AZURE_DI_ENDPOINT") or "").strip().rstrip("/")
AZURE_DI_KEY = (os.getenv("AZURE_DI_KEY") or "").strip()
AZURE_DI_API_VERSION = (os.getenv("AZURE_DI_API_VERSION") or "2024-11-30").strip()

DB_FILE = "edd_exp_azure.db"
UPLOAD_DIR = "uploaded_docs"

Path(UPLOAD_DIR).mkdir(exist_ok=True)

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
        max-width: 1550px;
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
    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .badge-red {
        display:inline-block;
        background:#b91c1c;
        color:white;
        padding:6px 12px;
        border-radius:999px;
        font-weight:600;
        margin-right:8px;
    }
    .badge-orange {
        display:inline-block;
        background:#c2410c;
        color:white;
        padding:6px 12px;
        border-radius:999px;
        font-weight:600;
        margin-right:8px;
    }
    .badge-green {
        display:inline-block;
        background:#15803d;
        color:white;
        padding:6px 12px;
        border-radius:999px;
        font-weight:600;
        margin-right:8px;
    }
    .small-note {
        opacity:0.8;
        font-size:0.88rem;
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
    if not s:
        return ""
    s = str(s).lower()
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

    s = str(value).strip()
    s = s.replace(".", "/").replace("-", "/")
    s = re.sub(r"\s+", "", s)

    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
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

def format_money(x):
    if x is None or pd.isna(x):
        return "-"
    try:
        return f"{float(x):,.2f}".replace(",", " ")
    except Exception:
        return str(x)

def format_pct(x):
    if x is None or pd.isna(x):
        return "-"
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)

def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)

def save_uploaded_file(uploaded_file):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = os.path.join(UPLOAD_DIR, f"{stamp}_{safe_filename(uploaded_file.name)}")
    with open(out, "wb") as f:
        f.write(uploaded_file.getvalue())
    return out

def infer_doc_type_from_name(name: str) -> str:
    n = normalize_for_matching(name)
    if "avoir" in n:
        return "facture_avoir"
    if "cotisation" in n:
        return "recu_cotisation"
    if "ordonnance" in n:
        return "ordonnance"
    return "facture"

def safe_div(a, b):
    try:
        if b in [0, None] or pd.isna(b):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def badge_html(label, level):
    if level == "red":
        return f'<span class="badge-red">{label}</span>'
    if level == "orange":
        return f'<span class="badge-orange">{label}</span>'
    return f'<span class="badge-green">{label}</span>'

def get_product_name(row):
    designation = normalize_text(row.get("designation"))
    reference = normalize_text(row.get("reference"))
    if designation:
        return designation
    if reference:
        return reference
    return "Produit non identifié"

# =========================================================
# DB
# =========================================================
def get_db_connection():
    con = sqlite3.connect(DB_FILE, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

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
            document_reference TEXT,
            document_date TEXT,
            due_date TEXT,
            period_start TEXT,
            period_end TEXT,
            client_name TEXT,
            patient_name TEXT,
            payment_mode TEXT,
            total_ht REAL,
            total_tva REAL,
            total_ttc REAL,
            currency TEXT,
            is_credit_note INTEGER DEFAULT 0,
            category TEXT,
            subcategory TEXT,
            cost_center TEXT,
            charge_nature TEXT,
            charge_behavior TEXT,
            raw_text TEXT,
            status_validation TEXT,
            azure_model_id TEXT,
            created_at TEXT
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
            lot_number TEXT,
            service_date TEXT,
            patient_name TEXT,
            quantity REAL,
            unit_price_ht REAL,
            unit_price_ttc REAL,
            discount REAL,
            vat_rate REAL,
            line_amount_ht REAL,
            line_amount_ttc REAL,
            cost_center TEXT,
            charge_category TEXT,
            raw_line TEXT,
            FOREIGN KEY(document_id) REFERENCES documents_achats(document_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS category_mappings (
            mapping_id TEXT PRIMARY KEY,
            supplier_keyword TEXT,
            designation_keyword TEXT,
            category TEXT,
            subcategory TEXT,
            cost_center TEXT,
            charge_nature TEXT,
            charge_behavior TEXT,
            priority INTEGER DEFAULT 100
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS suppliers (
            supplier_id TEXT PRIMARY KEY,
            supplier_name_raw TEXT,
            supplier_name_normalized TEXT,
            supplier_type TEXT,
            city TEXT,
            default_category TEXT,
            created_at TEXT
        )
        """
    )

    con.commit()

    cur.execute("SELECT COUNT(*) FROM category_mappings")
    count_map = cur.fetchone()[0]

    if count_map == 0:
        default_mappings = [
            ("pharmacie", "", "Médicaments", "Pharmacie", "Pharmacie", "Directe", "Variable", 10),
            ("boudor", "", "Médicaments", "Pharmacie", "Pharmacie", "Directe", "Variable", 5),
            ("ultimed", "", "Consommables médicaux", "Dispositifs médicaux", "Bloc / Soins", "Directe", "Variable", 5),
            ("medical", "", "Consommables médicaux", "Matériel médical", "Bloc / Soins", "Directe", "Variable", 20),
            ("transfusion", "", "Analyses & sang", "Transfusion", "Laboratoire / Soins", "Directe", "Variable", 5),
            ("sang", "", "Analyses & sang", "Produits sanguins", "Laboratoire / Soins", "Directe", "Variable", 1),
            ("agence marocaine du sang", "", "Analyses & sang", "Produits sanguins", "Laboratoire / Soins", "Directe", "Variable", 1),
            ("ordre national des medecins", "", "Charges administratives", "Cotisations professionnelles", "Administration", "Indirecte", "Fixe", 5),
            ("", "gel hydroalcoolique", "Hygiène", "Désinfection", "Soins / Hygiène", "Directe", "Variable", 5),
            ("", "stethoscope", "Consommables médicaux", "Petits équipements", "Bloc / Soins", "Directe", "Variable", 5),
            ("", "phenotype", "Analyses & sang", "Analyses", "Laboratoire", "Directe", "Variable", 5),
            ("", "abo rh", "Analyses & sang", "Analyses", "Laboratoire", "Directe", "Variable", 5),
            ("", "culot globulaire", "Analyses & sang", "Produits sanguins", "Laboratoire / Soins", "Directe", "Variable", 1),
            ("", "", "Autres charges", "Non classé", "À affecter", "À définir", "À définir", 999),
        ]
        for row in default_mappings:
            cur.execute(
                """
                INSERT INTO category_mappings (
                    mapping_id, supplier_keyword, designation_keyword, category,
                    subcategory, cost_center, charge_nature, charge_behavior, priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), *row),
            )
        con.commit()

    con.close()

init_db()

# =========================================================
# AZURE DOCUMENT INTELLIGENCE - REST
# =========================================================
def azure_is_configured():
    return bool(AZURE_DI_ENDPOINT and AZURE_DI_KEY)

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

# =========================================================
# NORMALISATION DU JSON AZURE
# =========================================================
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

def extract_agence_sang_lines_from_raw_text(raw_text: str):
    lines = []
    raw_lines = [x.strip() for x in raw_text.splitlines() if x.strip()]

    started = False
    for line in raw_lines:
        low = normalize_for_matching(line)

        if "nom & prenom" in low or "nom & prénom" in low:
            started = True
            continue

        if not started:
            continue

        if "arreter la presente facture" in low or "m.total" in low:
            break

        m = re.match(
            r"^(.*?)\s+(Culot\s+Globulaire\s*\(.*?\))\s+(\d{5,})\s+(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})\s+(\d+(?:[.,]\d+)?)\s+(\d+(?:[.,]\d+)?)\s+(\d+(?:[.,]\d+)?)$",
            line,
            flags=re.IGNORECASE,
        )

        if m:
            patient_name = normalize_text(m.group(1))
            designation = normalize_text(m.group(2))
            ref_bl = normalize_text(m.group(3))
            service_date = parse_date_fr(m.group(4))
            qty = parse_numeric_value(m.group(5))
            pu = parse_numeric_value(m.group(6))
            amount = parse_numeric_value(m.group(7))

            lines.append({
                "line_no": len(lines) + 1,
                "reference": ref_bl,
                "designation": designation,
                "lot_number": None,
                "service_date": service_date,
                "patient_name": patient_name,
                "quantity": qty,
                "unit_price_ht": pu,
                "unit_price_ttc": pu,
                "discount": np.nan,
                "vat_rate": np.nan,
                "line_amount_ht": amount,
                "line_amount_ttc": amount,
                "raw_line": line,
            })
    return lines

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
    purchase_order = field_to_text(get_field_obj(fields, ["PurchaseOrder", "PurchaseOrderNumber"]))
    payment_terms = field_to_text(get_field_obj(fields, ["PaymentTerm", "PaymentTerms"]))
    currency = (
        field_currency_code(get_field_obj(fields, ["AmountDue", "InvoiceTotal", "Total"]))
        or field_currency_code(get_field_obj(fields, ["SubTotal"]))
        or "MAD"
    )

    doc_type = infer_doc_type_from_name(filename)
    is_credit_note = 1 if "avoir" in normalize_for_matching(filename) else 0

    raw_norm = normalize_for_matching(raw_text)
    file_norm = normalize_for_matching(filename)

    if "agence marocaine du sang" in raw_norm or "agence marocaine du sang" in file_norm:
        vendor_name = "Agence Marocaine du Sang et de ses Dérivés"

    if not vendor_name and "agence marocaine du sang" in raw_norm:
        vendor_name = "Agence Marocaine du Sang et de ses Dérivés"

    if invoice_id:
        m = re.search(r"(\d{4}/\d{4,})", str(invoice_id))
        if m:
            invoice_id = m.group(1)
    else:
        m = re.search(r"(\d{4}/\d{4,})", raw_text)
        if m:
            invoice_id = m.group(1)

    if not customer_name:
        m = re.search(r"Client\s*:\s*(.+)", raw_text, flags=re.IGNORECASE)
        if m:
            customer_name = m.group(1).strip()

    period_start = None
    period_end = None
    m = re.search(
        r"Du\s*:\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})\s*Au\s*:\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})",
        raw_text,
        flags=re.IGNORECASE,
    )
    if m:
        period_start = parse_date_fr(m.group(1))
        period_end = parse_date_fr(m.group(2))

    if not invoice_date and period_start:
        invoice_date = period_start

    if "dirham" in raw_norm or "mad" in raw_norm or "sept mille deux cent dirhams" in raw_norm:
        currency = "MAD"
    if "agence marocaine du sang" in raw_norm:
        currency = "MAD"

    if pd.isna(amount_due):
        m = re.search(r"M\.?TOTAL\s*([0-9]+[.,]?[0-9]*)", raw_text, flags=re.IGNORECASE)
        if m:
            amount_due = parse_numeric_value(m.group(1))

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
            tax_value = field_to_number(get_field_obj(item_obj, ["Tax"]))

            lines.append({
                "line_no": idx,
                "reference": normalize_text(product_code),
                "designation": normalize_text(desc) or f"Ligne {idx}",
                "lot_number": None,
                "service_date": date_value,
                "patient_name": None,
                "quantity": qty,
                "unit_price_ht": unit_price,
                "unit_price_ttc": unit_price,
                "discount": np.nan,
                "vat_rate": tax_value if pd.notna(tax_value) else np.nan,
                "line_amount_ht": amount,
                "line_amount_ttc": amount,
                "raw_line": None,
            })

    if "agence marocaine du sang" in raw_norm:
        special_lines = extract_agence_sang_lines_from_raw_text(raw_text)
        if special_lines:
            lines = special_lines

    if not lines:
        lines = [{
            "line_no": 1,
            "reference": None,
            "designation": normalize_text(vendor_name) or "Facture",
            "lot_number": None,
            "service_date": invoice_date,
            "patient_name": None,
            "quantity": 1.0,
            "unit_price_ht": amount_due if pd.notna(amount_due) else np.nan,
            "unit_price_ttc": amount_due if pd.notna(amount_due) else np.nan,
            "discount": np.nan,
            "vat_rate": np.nan,
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
        "document_reference": normalize_text(purchase_order),
        "document_date": invoice_date,
        "due_date": due_date,
        "period_start": period_start,
        "period_end": period_end,
        "client_name": normalize_text(customer_name),
        "patient_name": None,
        "payment_mode": normalize_text(payment_terms),
        "total_ht": sub_total,
        "total_tva": total_tax,
        "total_ttc": amount_due,
        "currency": currency or "MAD",
        "is_credit_note": is_credit_note,
        "quantity_total": np.nansum([x.get("quantity", np.nan) for x in lines]),
        "unit_price_guess": np.nan,
        "raw_text": raw_text,
        "category": None,
        "subcategory": None,
        "cost_center": None,
        "charge_nature": None,
        "charge_behavior": None,
        "status_validation": "À revoir",
        "azure_model_id": analyze_result.get("modelId", "prebuilt-invoice"),
    }

    return doc, lines

# =========================================================
# CLASSIFICATION
# =========================================================
def load_mappings():
    con = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM category_mappings ORDER BY priority ASC", con)
    con.close()
    return df

def classify_document(supplier_name, raw_text, designation_sample=None):
    mappings = load_mappings()

    supplier_n = normalize_for_matching(supplier_name)
    text_n = normalize_for_matching(raw_text)
    designation_n = normalize_for_matching(designation_sample)

    for _, row in mappings.iterrows():
        sk = normalize_for_matching(row["supplier_keyword"])
        dk = normalize_for_matching(row["designation_keyword"])

        supplier_ok = (not sk) or (sk in supplier_n) or (sk in text_n)
        designation_ok = (not dk) or (dk in designation_n) or (dk in text_n)

        if supplier_ok and designation_ok:
            return {
                "category": row["category"],
                "subcategory": row["subcategory"],
                "cost_center": row["cost_center"],
                "charge_nature": row["charge_nature"],
                "charge_behavior": row["charge_behavior"],
            }

    return {
        "category": "Autres charges",
        "subcategory": "Non classé",
        "cost_center": "À affecter",
        "charge_nature": "À définir",
        "charge_behavior": "À définir",
    }

def classify_lines(lines, supplier_name, raw_text):
    out = []
    for row in lines:
        klass = classify_document(
            supplier_name=supplier_name,
            raw_text=raw_text,
            designation_sample=row.get("designation"),
        )
        row["cost_center"] = klass["cost_center"]
        row["charge_category"] = klass["category"]
        out.append(row)
    return out

# =========================================================
# EXTRACTION
# =========================================================
def extract_document_with_azure(uploaded_file, stored_path):
    if not azure_is_configured():
        raise RuntimeError("Azure DI non configuré. Vérifie AZURE_DI_ENDPOINT et AZURE_DI_KEY dans .env")

    file_bytes = uploaded_file.getvalue()
    result_json = azure_analyze_invoice(file_bytes, uploaded_file.name)
    doc, lines = normalize_azure_invoice_result(result_json, uploaded_file.name, stored_path)

    designation_sample = next((x.get("designation") for x in lines if x.get("designation")), "")
    klass = classify_document(doc["supplier_name"], doc["raw_text"], designation_sample)
    doc["category"] = klass["category"]
    doc["subcategory"] = klass["subcategory"]
    doc["cost_center"] = klass["cost_center"]
    doc["charge_nature"] = klass["charge_nature"]
    doc["charge_behavior"] = klass["charge_behavior"]

    lines = classify_lines(lines, doc["supplier_name"], doc["raw_text"])
    return doc, lines

# =========================================================
# DATAFRAMES FOR EDITING
# =========================================================
def build_doc_editor_df(doc):
    return pd.DataFrame([{
        "source_file": doc.get("source_file"),
        "doc_type": doc.get("doc_type"),
        "supplier_name": doc.get("supplier_name"),
        "invoice_number": doc.get("invoice_number"),
        "document_reference": doc.get("document_reference"),
        "document_date": doc.get("document_date"),
        "due_date": doc.get("due_date"),
        "period_start": doc.get("period_start"),
        "period_end": doc.get("period_end"),
        "client_name": doc.get("client_name"),
        "patient_name": doc.get("patient_name"),
        "payment_mode": doc.get("payment_mode"),
        "total_ht": doc.get("total_ht"),
        "total_tva": doc.get("total_tva"),
        "total_ttc": doc.get("total_ttc"),
        "currency": doc.get("currency"),
        "is_credit_note": doc.get("is_credit_note"),
        "category": doc.get("category"),
        "subcategory": doc.get("subcategory"),
        "cost_center": doc.get("cost_center"),
        "charge_nature": doc.get("charge_nature"),
        "charge_behavior": doc.get("charge_behavior"),
        "status_validation": doc.get("status_validation"),
        "azure_model_id": doc.get("azure_model_id"),
    }])

def build_lines_editor_df(lines):
    if not lines:
        return pd.DataFrame(columns=[
            "line_no", "reference", "designation", "lot_number", "service_date",
            "patient_name", "quantity", "unit_price_ht", "unit_price_ttc",
            "discount", "vat_rate", "line_amount_ht", "line_amount_ttc",
            "cost_center", "charge_category", "raw_line"
        ])
    return pd.DataFrame(lines)

# =========================================================
# SAVE / LOAD DB
# =========================================================
def upsert_supplier_reference(supplier_name, default_category=None):
    if not supplier_name:
        return

    con = get_db_connection()
    cur = con.cursor()

    norm = normalize_for_matching(supplier_name)
    cur.execute(
        """
        SELECT supplier_id FROM suppliers
        WHERE supplier_name_normalized = ?
        """,
        (norm,),
    )
    row = cur.fetchone()

    if not row:
        cur.execute(
            """
            INSERT INTO suppliers (
                supplier_id, supplier_name_raw, supplier_name_normalized,
                supplier_type, city, default_category, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                supplier_name,
                norm,
                None,
                None,
                default_category,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        con.commit()

    con.close()

def save_document_to_db(doc, lines_df):
    con = get_db_connection()
    cur = con.cursor()

    document_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    cur.execute(
        """
        INSERT INTO documents_achats (
            document_id, source_file, stored_file_path, doc_type, supplier_name,
            issuer_name, invoice_number, document_reference, document_date, due_date,
            period_start, period_end, client_name, patient_name, payment_mode,
            total_ht, total_tva, total_ttc, currency, is_credit_note,
            category, subcategory, cost_center, charge_nature, charge_behavior,
            raw_text, status_validation, azure_model_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            doc.get("source_file"),
            doc.get("stored_file_path"),
            doc.get("doc_type"),
            doc.get("supplier_name"),
            doc.get("issuer_name"),
            doc.get("invoice_number"),
            doc.get("document_reference"),
            doc.get("document_date"),
            doc.get("due_date"),
            doc.get("period_start"),
            doc.get("period_end"),
            doc.get("client_name"),
            doc.get("patient_name"),
            doc.get("payment_mode"),
            float(doc.get("total_ht")) if pd.notna(doc.get("total_ht")) else None,
            float(doc.get("total_tva")) if pd.notna(doc.get("total_tva")) else None,
            float(doc.get("total_ttc")) if pd.notna(doc.get("total_ttc")) else None,
            doc.get("currency"),
            int(doc.get("is_credit_note") or 0),
            doc.get("category"),
            doc.get("subcategory"),
            doc.get("cost_center"),
            doc.get("charge_nature"),
            doc.get("charge_behavior"),
            doc.get("raw_text"),
            doc.get("status_validation"),
            doc.get("azure_model_id"),
            now,
        ),
    )

    if lines_df is not None and not lines_df.empty:
        for _, row in lines_df.iterrows():
            cur.execute(
                """
                INSERT INTO document_lines (
                    line_id, document_id, source_file, line_no, reference,
                    designation, lot_number, service_date, patient_name,
                    quantity, unit_price_ht, unit_price_ttc, discount, vat_rate,
                    line_amount_ht, line_amount_ttc, cost_center, charge_category, raw_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    document_id,
                    doc.get("source_file"),
                    int(row["line_no"]) if pd.notna(row.get("line_no")) else None,
                    normalize_text(row.get("reference")),
                    normalize_text(row.get("designation")),
                    normalize_text(row.get("lot_number")),
                    parse_date_fr(row.get("service_date")) if row.get("service_date") else None,
                    normalize_text(row.get("patient_name")),
                    float(row["quantity"]) if pd.notna(row.get("quantity")) else None,
                    float(row["unit_price_ht"]) if pd.notna(row.get("unit_price_ht")) else None,
                    float(row["unit_price_ttc"]) if pd.notna(row.get("unit_price_ttc")) else None,
                    float(row["discount"]) if pd.notna(row.get("discount")) else None,
                    float(row["vat_rate"]) if pd.notna(row.get("vat_rate")) else None,
                    float(row["line_amount_ht"]) if pd.notna(row.get("line_amount_ht")) else None,
                    float(row["line_amount_ttc"]) if pd.notna(row.get("line_amount_ttc")) else None,
                    normalize_text(row.get("cost_center")),
                    normalize_text(row.get("charge_category")),
                    normalize_text(row.get("raw_line")),
                ),
            )

    con.commit()
    con.close()
    upsert_supplier_reference(doc.get("supplier_name"), doc.get("category"))
    return document_id

def delete_document(document_id):
    con = get_db_connection()
    cur = con.cursor()
    cur.execute("DELETE FROM document_lines WHERE document_id = ?", (document_id,))
    cur.execute("DELETE FROM documents_achats WHERE document_id = ?", (document_id,))
    con.commit()
    con.close()

def load_all_data():
    con = get_db_connection()
    docs = pd.read_sql_query("SELECT * FROM documents_achats", con)
    lines = pd.read_sql_query("SELECT * FROM document_lines", con)
    mappings = pd.read_sql_query("SELECT * FROM category_mappings ORDER BY priority ASC", con)
    suppliers = pd.read_sql_query("SELECT * FROM suppliers", con)
    con.close()

    if not docs.empty:
        for c in ["document_date", "due_date", "period_start", "period_end", "created_at"]:
            if c in docs.columns:
                docs[c] = pd.to_datetime(docs[c], errors="coerce")
        for c in ["total_ht", "total_tva", "total_ttc"]:
            if c in docs.columns:
                docs[c] = pd.to_numeric(docs[c], errors="coerce")
        docs["month"] = docs["document_date"].dt.to_period("M").astype(str)
        docs["year"] = docs["document_date"].dt.year
        docs["month_date"] = pd.to_datetime(docs["month"].astype(str) + "-01", errors="coerce")

    if not lines.empty:
        for c in ["service_date"]:
            if c in lines.columns:
                lines[c] = pd.to_datetime(lines[c], errors="coerce")
        for c in ["quantity", "unit_price_ht", "unit_price_ttc", "discount", "vat_rate", "line_amount_ht", "line_amount_ttc"]:
            if c in lines.columns:
                lines[c] = pd.to_numeric(lines[c], errors="coerce")
        lines["product_name"] = lines.apply(get_product_name, axis=1)

    return docs, lines, mappings, suppliers

# =========================================================
# KPI / ANALYTICS / AUDIT
# =========================================================
def compute_kpis(docs_df):
    if docs_df.empty:
        return {
            "total_docs": 0,
            "total_amount": 0.0,
            "suppliers": 0,
            "avg_doc": 0.0,
            "credit_notes": 0,
            "top_supplier_share": 0.0,
        }

    total_amount = docs_df["total_ttc"].fillna(0).sum()
    supplier_sum = docs_df.groupby("supplier_name", dropna=False)["total_ttc"].sum().sort_values(ascending=False)
    top_share = (supplier_sum.iloc[0] / total_amount) if len(supplier_sum) and total_amount != 0 else 0

    return {
        "total_docs": int(len(docs_df)),
        "total_amount": float(total_amount),
        "suppliers": int(docs_df["supplier_name"].nunique(dropna=True)),
        "avg_doc": float(docs_df["total_ttc"].mean()) if len(docs_df) else 0.0,
        "credit_notes": int((docs_df["is_credit_note"] == 1).sum()),
        "top_supplier_share": float(top_share),
    }

def assign_abc_class(total_by_supplier):
    if total_by_supplier.empty:
        return pd.DataFrame(columns=["supplier_name", "total_ttc", "share", "cum_share", "abc_class"])
    df = total_by_supplier.sort_values("total_ttc", ascending=False).copy()
    total = df["total_ttc"].sum()
    df["share"] = np.where(total > 0, df["total_ttc"] / total, 0)
    df["cum_share"] = df["share"].cumsum()

    def classify(x):
        if x <= 0.80:
            return "A"
        elif x <= 0.95:
            return "B"
        return "C"

    df["abc_class"] = df["cum_share"].apply(classify)
    return df

def prepare_filtered_data(docs_df, lines_df, supplier_filter, category_filter, doc_type_filter,
                          selected_periods, amount_min, amount_max):
    view_docs = docs_df.copy()

    if supplier_filter:
        view_docs = view_docs[view_docs["supplier_name"].isin(supplier_filter)]
    if category_filter:
        view_docs = view_docs[view_docs["category"].isin(category_filter)]
    if doc_type_filter:
        view_docs = view_docs[view_docs["doc_type"].isin(doc_type_filter)]
    if selected_periods:
        view_docs = view_docs[view_docs["month"].isin(selected_periods)]

    view_docs = view_docs[
        (view_docs["total_ttc"].fillna(0) >= amount_min) &
        (view_docs["total_ttc"].fillna(0) <= amount_max)
    ]

    if lines_df.empty or view_docs.empty:
        view_lines = pd.DataFrame()
    else:
        view_lines = lines_df.merge(
            view_docs[["document_id", "supplier_name", "month", "document_date", "category", "doc_type"]],
            on="document_id",
            how="inner",
        )

    return view_docs, view_lines

def compute_supplier_metrics(view_docs, monthly_ca=None, total_patients=None):
    if view_docs.empty:
        return {}

    total_spend = view_docs["total_ttc"].fillna(0).sum()

    supplier_totals = (
        view_docs.groupby("supplier_name", dropna=False)["total_ttc"]
        .sum()
        .reset_index()
        .sort_values("total_ttc", ascending=False)
    )

    top3_share = safe_div(supplier_totals.head(3)["total_ttc"].sum(), total_spend)
    ticket_moyen = view_docs["total_ttc"].mean()
    cout_moyen_fournisseur = supplier_totals["total_ttc"].mean()

    monthly_invoice_count = (
        view_docs.groupby("month")["document_id"]
        .nunique()
        .reset_index(name="nb_factures")
        .sort_values("month")
    )
    frequence_achat = monthly_invoice_count["nb_factures"].mean() if not monthly_invoice_count.empty else np.nan

    monthly_spend = (
        view_docs.groupby("month")["total_ttc"]
        .sum()
        .reset_index()
        .sort_values("month")
    )
    monthly_spend["variation_pct"] = monthly_spend["total_ttc"].pct_change()

    top_supplier = supplier_totals.iloc[0]["supplier_name"] if not supplier_totals.empty else None
    top_supplier_share = safe_div(supplier_totals.iloc[0]["total_ttc"], total_spend) if not supplier_totals.empty else np.nan

    supplier_ca_ratio = safe_div(total_spend, monthly_ca) if monthly_ca and monthly_ca > 0 else np.nan
    cout_par_patient = safe_div(total_spend, total_patients) if total_patients and total_patients > 0 else np.nan

    return {
        "total_spend": total_spend,
        "supplier_totals": supplier_totals,
        "top3_share": top3_share,
        "ticket_moyen": ticket_moyen,
        "cout_moyen_fournisseur": cout_moyen_fournisseur,
        "frequence_achat": frequence_achat,
        "monthly_spend": monthly_spend,
        "top_supplier": top_supplier,
        "top_supplier_share": top_supplier_share,
        "supplier_ca_ratio": supplier_ca_ratio,
        "cout_par_patient": cout_par_patient,
    }

def compute_charge_split(view_docs):
    if view_docs.empty:
        return pd.DataFrame(columns=["type_charge", "montant"])

    df = view_docs.copy()
    df["charge_behavior_clean"] = df["charge_behavior"].fillna("À définir")
    fixed_amount = df.loc[df["charge_behavior_clean"].str.lower().eq("fixe"), "total_ttc"].fillna(0).sum()
    variable_amount = df.loc[df["charge_behavior_clean"].str.lower().eq("variable"), "total_ttc"].fillna(0).sum()
    other_amount = df.loc[~df["charge_behavior_clean"].str.lower().isin(["fixe", "variable"]), "total_ttc"].fillna(0).sum()

    return pd.DataFrame({
        "type_charge": ["Charges fixes", "Charges variables", "Autres / à définir"],
        "montant": [fixed_amount, variable_amount, other_amount]
    })

def compute_category_breakdown(view_docs):
    if view_docs.empty:
        return pd.DataFrame(columns=["category", "total_ttc"])
    df = (
        view_docs.groupby("category", dropna=False)["total_ttc"]
        .sum()
        .reset_index()
        .sort_values("total_ttc", ascending=False)
    )
    return df

def compute_heatmap_month_category(view_docs):
    if view_docs.empty:
        return pd.DataFrame()
    pivot = pd.pivot_table(
        view_docs,
        values="total_ttc",
        index="category",
        columns="month",
        aggfunc="sum",
        fill_value=0
    )
    return pivot

def compute_cost_center_breakdown(view_docs):
    if view_docs.empty:
        return pd.DataFrame(columns=["cost_center", "total_ttc"])
    return (
        view_docs.groupby("cost_center", dropna=False)["total_ttc"]
        .sum()
        .reset_index()
        .sort_values("total_ttc", ascending=False)
    )

def compute_supplier_strategic_table(view_docs):
    if view_docs.empty:
        return pd.DataFrame()

    total_spend = view_docs["total_ttc"].fillna(0).sum()
    supplier_stats = (
        view_docs.groupby("supplier_name", dropna=False)
        .agg(
            total_ttc=("total_ttc", "sum"),
            nb_factures=("document_id", "nunique"),
            avg_ticket=("total_ttc", "mean"),
            std_ticket=("total_ttc", "std"),
            nb_mois=("month", "nunique"),
        )
        .reset_index()
    )

    supplier_stats["dependance_pct"] = np.where(total_spend > 0, supplier_stats["total_ttc"] / total_spend, 0)

    abc_df = assign_abc_class(supplier_stats[["supplier_name", "total_ttc"]].copy())
    supplier_stats = supplier_stats.merge(
        abc_df[["supplier_name", "abc_class"]],
        on="supplier_name",
        how="left"
    )

    supplier_stats["freq_score"] = np.where(supplier_stats["nb_factures"].max() > 0, supplier_stats["nb_factures"] / supplier_stats["nb_factures"].max(), 0)
    supplier_stats["volume_score"] = np.where(supplier_stats["total_ttc"].max() > 0, supplier_stats["total_ttc"] / supplier_stats["total_ttc"].max(), 0)

    supplier_stats["price_stability_score"] = 1 - np.where(
        supplier_stats["avg_ticket"].fillna(0) > 0,
        (supplier_stats["std_ticket"].fillna(0) / supplier_stats["avg_ticket"].replace(0, np.nan)).clip(upper=1),
        1
    )
    supplier_stats["price_stability_score"] = supplier_stats["price_stability_score"].fillna(0).clip(lower=0, upper=1)

    supplier_stats["supplier_score"] = (
        supplier_stats["volume_score"] * 0.5 +
        supplier_stats["freq_score"] * 0.3 +
        supplier_stats["price_stability_score"] * 0.2
    ) * 100

    supplier_stats = supplier_stats.sort_values("total_ttc", ascending=False)
    return supplier_stats

def compute_product_analytics(view_lines):
    if view_lines.empty:
        return {
            "product_summary": pd.DataFrame(),
            "monthly_price": pd.DataFrame(),
            "supplier_price_compare": pd.DataFrame(),
            "inflation_table": pd.DataFrame(),
        }

    df = view_lines.copy()
    df["unit_price"] = df["unit_price_ttc"].fillna(df["unit_price_ht"])
    df["line_amount"] = df["line_amount_ttc"].fillna(df["line_amount_ht"])
    df["product_name"] = df["product_name"].fillna("Produit non identifié")

    product_summary = (
        df.groupby("product_name")
        .agg(
            quantite_totale=("quantity", "sum"),
            cout_total=("line_amount", "sum"),
            cout_moyen=("line_amount", "mean"),
            prix_moyen=("unit_price", "mean"),
        )
        .reset_index()
    )

    monthly_price = (
        df.groupby(["month", "product_name"])["unit_price"]
        .mean()
        .reset_index()
        .sort_values(["product_name", "month"])
    )

    supplier_price_compare = (
        df.groupby(["product_name", "supplier_name"])["unit_price"]
        .mean()
        .reset_index()
        .sort_values(["product_name", "unit_price"], ascending=[True, False])
    )

    inflation_table = monthly_price.copy()
    inflation_table["previous_price"] = inflation_table.groupby("product_name")["unit_price"].shift(1)
    inflation_table["variation_pct"] = np.where(
        inflation_table["previous_price"].fillna(0) > 0,
        (inflation_table["unit_price"] - inflation_table["previous_price"]) / inflation_table["previous_price"],
        np.nan
    )

    return {
        "product_summary": product_summary,
        "monthly_price": monthly_price,
        "supplier_price_compare": supplier_price_compare,
        "inflation_table": inflation_table,
    }

def build_audit_alerts(docs_df, lines_df):
    alerts = []

    if docs_df.empty:
        return pd.DataFrame(alerts)

    docs = docs_df.copy()
    docs["total_ttc"] = pd.to_numeric(docs["total_ttc"], errors="coerce")

    dup_combo = docs[
        docs["supplier_name"].fillna("").astype(str).str.strip().ne("") &
        docs["document_date"].notna() &
        docs["total_ttc"].notna()
    ].copy()

    if not dup_combo.empty:
        dup_combo["dup_key"] = (
            dup_combo["supplier_name"].fillna("").astype(str).str.lower().str.strip() + "|" +
            dup_combo["document_date"].astype(str) + "|" +
            dup_combo["total_ttc"].round(2).astype(str)
        )
        dup_rows = dup_combo[dup_combo["dup_key"].duplicated(keep=False)]
        for _, row in dup_rows.iterrows():
            alerts.append({
                "type_controle": "Facture potentiellement dupliquée",
                "niveau_risque": "Élevé",
                "supplier_name": row.get("supplier_name"),
                "reference": row.get("invoice_number"),
                "observation": "Même fournisseur + même date + même montant",
                "valeur": row.get("total_ttc"),
                "badge": "red",
            })

    if "invoice_number" in docs.columns:
        dup_num = docs[
            docs["invoice_number"].fillna("").astype(str).str.strip().ne("") &
            docs["invoice_number"].fillna("").duplicated(keep=False)
        ]
        for _, row in dup_num.iterrows():
            alerts.append({
                "type_controle": "Doublon numéro facture",
                "niveau_risque": "Élevé",
                "supplier_name": row.get("supplier_name"),
                "reference": row.get("invoice_number"),
                "observation": "Même numéro détecté plusieurs fois",
                "valeur": row.get("total_ttc"),
                "badge": "red",
            })

    q95 = docs["total_ttc"].quantile(0.95) if docs["total_ttc"].notna().any() else np.nan
    extreme_threshold = docs["total_ttc"].mean() + 2 * docs["total_ttc"].std() if docs["total_ttc"].notna().any() else np.nan

    if pd.notna(extreme_threshold):
        high_amount = docs[docs["total_ttc"] > extreme_threshold]
        for _, row in high_amount.iterrows():
            alerts.append({
                "type_controle": "Montant anormalement élevé",
                "niveau_risque": "Élevé" if pd.notna(q95) and row["total_ttc"] > q95 else "Moyen",
                "supplier_name": row.get("supplier_name"),
                "reference": row.get("invoice_number"),
                "observation": "Montant très supérieur aux autres factures",
                "valeur": row.get("total_ttc"),
                "badge": "red" if pd.notna(q95) and row["total_ttc"] > q95 else "orange",
            })

    zero_total = docs[pd.to_numeric(docs["total_ttc"], errors="coerce").fillna(0) == 0]
    for _, row in zero_total.iterrows():
        alerts.append({
            "type_controle": "Montant nul ou absent",
            "niveau_risque": "Élevé",
            "supplier_name": row.get("supplier_name"),
            "reference": row.get("invoice_number"),
            "observation": "Total TTC nul ou non extrait",
            "valeur": row.get("total_ttc"),
            "badge": "red",
        })

    miss_core = docs[
        docs["supplier_name"].isna() |
        docs["document_date"].isna() |
        docs["total_ttc"].isna() |
        docs["invoice_number"].isna()
    ]
    for _, row in miss_core.iterrows():
        alerts.append({
            "type_controle": "Facture avec informations manquantes",
            "niveau_risque": "Moyen",
            "supplier_name": row.get("supplier_name"),
            "reference": row.get("invoice_number"),
            "observation": "Certaines données clés sont manquantes",
            "valeur": row.get("total_ttc"),
            "badge": "orange",
        })

    if not lines_df.empty:
        check = lines_df.copy()
        check["calc_amount"] = check["quantity"] * check["unit_price_ttc"]
        check["delta"] = (check["line_amount_ttc"] - check["calc_amount"]).abs()
        incoherent = check[
            pd.notna(check["quantity"]) &
            pd.notna(check["unit_price_ttc"]) &
            pd.notna(check["line_amount_ttc"]) &
            (check["delta"] > 5)
        ]

        for _, row in incoherent.head(300).iterrows():
            alerts.append({
                "type_controle": "Prix unitaire incohérent",
                "niveau_risque": "Moyen",
                "supplier_name": None,
                "reference": row.get("designation"),
                "observation": "Montant ligne différent du calcul Qté x PU",
                "valeur": row.get("line_amount_ttc"),
                "badge": "orange",
            })

        qty_threshold = check["quantity"].mean() + 2 * check["quantity"].std() if check["quantity"].notna().any() else np.nan
        if pd.notna(qty_threshold):
            abnormal_qty = check[check["quantity"] > qty_threshold]
            for _, row in abnormal_qty.head(300).iterrows():
                alerts.append({
                    "type_controle": "Quantité anormale",
                    "niveau_risque": "Moyen",
                    "supplier_name": None,
                    "reference": row.get("designation"),
                    "observation": "Quantité significativement supérieure à la normale",
                    "valeur": row.get("quantity"),
                    "badge": "orange",
                })

    total_spend = docs["total_ttc"].fillna(0).sum()
    supplier_sum = docs.groupby("supplier_name")["total_ttc"].sum().sort_values(ascending=False).reset_index()
    if not supplier_sum.empty and total_spend > 0:
        supplier_sum["share"] = supplier_sum["total_ttc"] / total_spend
        dominant = supplier_sum[supplier_sum["share"] > 0.40]
        for _, row in dominant.iterrows():
            alerts.append({
                "type_controle": "Fournisseur dominant",
                "niveau_risque": "Élevé",
                "supplier_name": row.get("supplier_name"),
                "reference": None,
                "observation": f"Dépendance > 40% ({row['share']:.1%})",
                "valeur": row.get("total_ttc"),
                "badge": "red",
            })

    monthly_spend = docs.groupby("month")["total_ttc"].sum().reset_index().sort_values("month")
    if not monthly_spend.empty:
        monthly_spend["variation_pct"] = monthly_spend["total_ttc"].pct_change()
        explosion = monthly_spend[monthly_spend["variation_pct"] > 0.30]
        for _, row in explosion.iterrows():
            alerts.append({
                "type_controle": "Explosion des dépenses",
                "niveau_risque": "Élevé",
                "supplier_name": None,
                "reference": row.get("month"),
                "observation": f"Variation mensuelle de {row['variation_pct']:.1%}",
                "valeur": row.get("total_ttc"),
                "badge": "red",
            })

    out = pd.DataFrame(alerts)
    if not out.empty:
        out = out.drop_duplicates(subset=["type_controle", "supplier_name", "reference", "observation", "valeur"])
    return out

def generate_auto_analysis(view_docs, view_lines, supplier_metrics, strategic_df, anomalies_df):
    messages = []

    if view_docs.empty:
        return ["Aucune donnée disponible pour générer une analyse automatique."]

    total_spend = supplier_metrics.get("total_spend", 0)
    top3_share = supplier_metrics.get("top3_share", np.nan)
    top_supplier = supplier_metrics.get("top_supplier")
    top_supplier_share = supplier_metrics.get("top_supplier_share", np.nan)

    if pd.notna(top_supplier_share):
        if top_supplier_share >= 0.70:
            messages.append(f"Les dépenses sont très fortement concentrées chez 1 fournisseur ({top_supplier_share:.0%}) : {top_supplier}. Le risque de dépendance est élevé.")
        elif top_supplier_share >= 0.40:
            messages.append(f"Le fournisseur principal {top_supplier} représente {top_supplier_share:.0%} des achats. Une surveillance de la dépendance est recommandée.")
        else:
            messages.append(f"La dépendance au premier fournisseur reste modérée ({top_supplier_share:.0%}).")

    if pd.notna(top3_share):
        if top3_share >= 0.80:
            messages.append(f"Les top 3 fournisseurs concentrent {top3_share:.0%} des dépenses, ce qui traduit une concentration élevée du portefeuille achats.")
        elif top3_share >= 0.60:
            messages.append(f"Les top 3 fournisseurs pèsent {top3_share:.0%} des dépenses. La concentration est notable.")
        else:
            messages.append(f"La concentration fournisseurs sur les top 3 reste contenue à {top3_share:.0%}.")

    monthly_spend = supplier_metrics.get("monthly_spend", pd.DataFrame())
    if not monthly_spend.empty and len(monthly_spend) >= 2:
        last_var = monthly_spend["variation_pct"].dropna()
        if not last_var.empty:
            v = last_var.iloc[-1]
            if v > 0.25:
                messages.append(f"Les dépenses du dernier mois affichent une hausse marquée de {v:.0%}.")
            elif v < -0.20:
                messages.append(f"Les dépenses du dernier mois reculent de {abs(v):.0%}, à analyser pour identifier la cause.")
            else:
                messages.append(f"La variation mensuelle récente reste globalement maîtrisée ({v:.0%}).")

    if not view_docs.empty and "category" in view_docs.columns:
        cat = (
            view_docs.groupby("category")["total_ttc"]
            .sum()
            .sort_values(ascending=False)
        )
        if not cat.empty:
            dominant_cat = cat.index[0]
            dominant_share = cat.iloc[0] / cat.sum() if cat.sum() > 0 else np.nan
            if pd.notna(dominant_share):
                messages.append(f"La catégorie dominante est '{dominant_cat}' avec {dominant_share:.0%} du total des dépenses.")

    if not anomalies_df.empty:
        high_risk = (anomalies_df["niveau_risque"] == "Élevé").sum()
        medium_risk = (anomalies_df["niveau_risque"] == "Moyen").sum()
        messages.append(f"{len(anomalies_df)} anomalies ont été détectées, dont {high_risk} à risque élevé et {medium_risk} à risque moyen.")

    if not strategic_df.empty:
        weak_stability = strategic_df[strategic_df["price_stability_score"] < 0.4]
        if not weak_stability.empty:
            supplier = weak_stability.sort_values("price_stability_score").iloc[0]["supplier_name"]
            messages.append(f"Le fournisseur {supplier} présente une stabilité de prix faible. Une revue des conditions tarifaires est conseillée.")

    if not view_lines.empty:
        unit_price_df = view_lines.copy()
        unit_price_df["unit_price"] = unit_price_df["unit_price_ttc"].fillna(unit_price_df["unit_price_ht"])
        if unit_price_df["unit_price"].notna().sum() > 0:
            high_price = unit_price_df["unit_price"].quantile(0.95)
            high_rows = unit_price_df[unit_price_df["unit_price"] >= high_price]
            if not high_rows.empty:
                prod = high_rows.iloc[0]["product_name"]
                messages.append(f"Le produit '{prod}' figure parmi les prix unitaires les plus élevés et mérite une comparaison fournisseurs.")

    return messages[:8]

def build_excel_report(view_docs, view_lines, anomalies_df, strategic_df, product_pack):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        view_docs.to_excel(writer, sheet_name="Documents filtrés", index=False)
        if not view_lines.empty:
            view_lines.to_excel(writer, sheet_name="Lignes filtrées", index=False)
        if not anomalies_df.empty:
            anomalies_df.to_excel(writer, sheet_name="Anomalies", index=False)
        if not strategic_df.empty:
            strategic_df.to_excel(writer, sheet_name="Strategie fournisseurs", index=False)

        if product_pack.get("product_summary") is not None and not product_pack["product_summary"].empty:
            product_pack["product_summary"].to_excel(writer, sheet_name="Produits", index=False)
        if product_pack.get("monthly_price") is not None and not product_pack["monthly_price"].empty:
            product_pack["monthly_price"].to_excel(writer, sheet_name="Prix mensuels", index=False)
        if product_pack.get("supplier_price_compare") is not None and not product_pack["supplier_price_compare"].empty:
            product_pack["supplier_price_compare"].to_excel(writer, sheet_name="Comparatif prix", index=False)

    output.seek(0)
    return output.getvalue()

def build_pdf_report(view_docs, anomalies_df, auto_messages):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Rapport d’analyse fournisseurs", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Généré le : {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    if not view_docs.empty:
        total_spend = view_docs["total_ttc"].fillna(0).sum()
        nb_docs = len(view_docs)
        nb_suppliers = view_docs["supplier_name"].nunique()
        summary_data = [
            ["Indicateur", "Valeur"],
            ["Nombre de documents", str(nb_docs)],
            ["Nombre de fournisseurs", str(nb_suppliers)],
            ["Montant total TTC", format_money(total_spend)],
        ]
        table = Table(summary_data, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f3d7a")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(table)
        story.append(Spacer(1, 14))

    story.append(Paragraph("Analyse automatique", styles["Heading2"]))
    for msg in auto_messages:
        story.append(Paragraph(f"• {msg}", styles["Normal"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Anomalies principales", styles["Heading2"]))

    if anomalies_df.empty:
        story.append(Paragraph("Aucune anomalie détectée.", styles["Normal"]))
    else:
        top_anom = anomalies_df.head(20).copy()
        rows = [["Type", "Risque", "Fournisseur", "Référence", "Observation", "Valeur"]]
        for _, r in top_anom.iterrows():
            rows.append([
                str(r.get("type_controle", "")),
                str(r.get("niveau_risque", "")),
                str(r.get("supplier_name", "") or ""),
                str(r.get("reference", "") or ""),
                str(r.get("observation", "") or ""),
                format_money(r.get("valeur")),
            ])

        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f3d7a")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
            ("PADDING", (0,0), (-1,-1), 4),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))
        story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# =========================================================
# LOAD GLOBAL DATA
# =========================================================
docs_df, lines_df, mappings_df, suppliers_df = load_all_data()
kpis = compute_kpis(docs_df)
audit_df = build_audit_alerts(docs_df, lines_df)

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
        "Analyse",
        "Audit",
        "Mappings analytiques",
        "Référentiel fournisseurs",
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
    st.write("Extraction robuste via Azure Document Intelligence + base documentaire + analyse avancée fournisseurs.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Documents", str(kpis["total_docs"]), "Base enregistrée")
    with c2:
        metric_card("Montant total TTC", format_money(kpis["total_amount"]), "Net des avoirs")
    with c3:
        metric_card("Fournisseurs", str(kpis["suppliers"]), "Référentiel détecté")
    with c4:
        metric_card("Avoirs", str(kpis["credit_notes"]), "Documents de type avoir")

    c5, c6, c7 = st.columns(3)
    with c5:
        metric_card("Document moyen", format_money(kpis["avg_doc"]))
    with c6:
        metric_card("Part top fournisseur", format_pct(kpis["top_supplier_share"]))
    with c7:
        metric_card("Alertes audit", str(len(audit_df)))

    st.markdown("---")
    st.write(f"Azure configuré : {'Oui' if azure_is_configured() else 'Non'}")
    if azure_is_configured():
        st.write(f"Endpoint : {AZURE_DI_ENDPOINT}")
        st.write(f"API version : {AZURE_DI_API_VERSION}")
    else:
        st.warning("Renseigne AZURE_DI_ENDPOINT et AZURE_DI_KEY dans .env")

# =========================================================
# PAGE IMPORT
# =========================================================
elif menu == "Import & Extraction":
    st.subheader("Import des documents")
    st.info("L'extraction est faite par Azure Document Intelligence (modèle facture).")

    files = st.file_uploader(
        "Télécharger un ou plusieurs documents",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        accept_multiple_files=True,
    )

    if not azure_is_configured():
        st.error("Azure DI n'est pas configuré. Ajoute AZURE_DI_ENDPOINT et AZURE_DI_KEY dans .env")
        st.stop()

    if not files:
        st.info("Ajoute des fichiers pour lancer l’extraction.")
    else:
        for idx, file in enumerate(files, start=1):
            st.markdown(f"### Document {idx} - {file.name}")
            stored_path = save_uploaded_file(file)

            with st.spinner(f"Analyse Azure de {file.name}..."):
                try:
                    doc, lines = extract_document_with_azure(file, stored_path)
                except Exception as e:
                    st.error(f"Échec d'analyse pour {file.name} : {e}")
                    continue

            doc_editor = build_doc_editor_df(doc)
            lines_editor = build_lines_editor_df(lines)

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

            if st.button(f"Enregistrer {file.name}", use_container_width=True, key=f"save_btn_{idx}"):
                row0 = edited_doc.iloc[0]

                doc_to_save = {
                    "source_file": file.name,
                    "stored_file_path": stored_path,
                    "doc_type": normalize_text(row0.get("doc_type")),
                    "supplier_name": normalize_text(row0.get("supplier_name")),
                    "issuer_name": normalize_text(row0.get("supplier_name")),
                    "invoice_number": normalize_text(row0.get("invoice_number")),
                    "document_reference": normalize_text(row0.get("document_reference")),
                    "document_date": parse_date_fr(row0.get("document_date")) if row0.get("document_date") else None,
                    "due_date": parse_date_fr(row0.get("due_date")) if row0.get("due_date") else None,
                    "period_start": parse_date_fr(row0.get("period_start")) if row0.get("period_start") else None,
                    "period_end": parse_date_fr(row0.get("period_end")) if row0.get("period_end") else None,
                    "client_name": normalize_text(row0.get("client_name")),
                    "patient_name": normalize_text(row0.get("patient_name")),
                    "payment_mode": normalize_text(row0.get("payment_mode")),
                    "total_ht": parse_numeric_value(row0.get("total_ht")),
                    "total_tva": parse_numeric_value(row0.get("total_tva")),
                    "total_ttc": parse_numeric_value(row0.get("total_ttc")),
                    "currency": normalize_text(row0.get("currency")) or "MAD",
                    "is_credit_note": int(parse_numeric_value(row0.get("is_credit_note")) or 0),
                    "category": normalize_text(row0.get("category")),
                    "subcategory": normalize_text(row0.get("subcategory")),
                    "cost_center": normalize_text(row0.get("cost_center")),
                    "charge_nature": normalize_text(row0.get("charge_nature")),
                    "charge_behavior": normalize_text(row0.get("charge_behavior")),
                    "raw_text": doc.get("raw_text", ""),
                    "status_validation": normalize_text(row0.get("status_validation")) or "À revoir",
                    "azure_model_id": normalize_text(row0.get("azure_model_id")) or "prebuilt-invoice",
                }

                if not edited_lines.empty:
                    for c in ["quantity", "unit_price_ht", "unit_price_ttc", "discount", "vat_rate", "line_amount_ht", "line_amount_ttc"]:
                        if c in edited_lines.columns:
                            edited_lines[c] = edited_lines[c].apply(parse_numeric_value)

                doc_id = save_document_to_db(doc_to_save, edited_lines)
                st.success(f"Document enregistré avec succès. ID : {doc_id}")
                st.rerun()

# =========================================================
# PAGE BASE DOCUMENTS
# =========================================================
elif menu == "Base Documents":
    st.subheader("Base documents")

    docs_df, lines_df, mappings_df, suppliers_df = load_all_data()

    if docs_df.empty:
        st.info("Aucun document enregistré.")
        st.stop()

    show = docs_df.copy()
    for c in ["total_ht", "total_tva", "total_ttc"]:
        if c in show.columns:
            show[c] = show[c].apply(format_money)

    cols = [
        "document_id", "source_file", "doc_type", "supplier_name", "invoice_number",
        "document_date", "due_date", "period_start", "period_end",
        "client_name", "payment_mode",
        "total_ht", "total_tva", "total_ttc", "currency",
        "is_credit_note", "category", "subcategory", "cost_center",
        "charge_nature", "charge_behavior", "status_validation", "azure_model_id", "created_at"
    ]
    cols = [c for c in cols if c in show.columns]

    st.dataframe(show[cols], use_container_width=True, hide_index=True)

    st.markdown("### Supprimer une facture")

    docs_df = docs_df.copy()
    docs_df["label_delete"] = (
        docs_df["source_file"].fillna("").astype(str)
        + " | "
        + docs_df["supplier_name"].fillna("").astype(str)
        + " | "
        + docs_df["invoice_number"].fillna("").astype(str)
    )

    selected_label = st.selectbox(
        "Choisir la facture à supprimer",
        docs_df["label_delete"].tolist(),
        key="delete_invoice_select"
    )

    selected_row = docs_df.loc[docs_df["label_delete"] == selected_label].iloc[0]

    st.warning("Cette suppression enlève aussi toutes les lignes détaillées liées à la facture.")
    confirm_delete = st.checkbox("Je confirme la suppression", key="confirm_delete_invoice")

    if st.button("Supprimer définitivement cette facture", use_container_width=True):
        if confirm_delete:
            delete_document(selected_row["document_id"])
            st.success("Facture supprimée avec succès.")
            st.rerun()
        else:
            st.error("Coche d’abord la confirmation de suppression.")

# =========================================================
# PAGE ANALYSE
# =========================================================
elif menu == "Analyse":
    st.subheader("Analyse fournisseurs, analytique et stratégique")

    docs_df, lines_df, mappings_df, suppliers_df = load_all_data()

    if docs_df.empty:
        st.info("Aucun document enregistré.")
        st.stop()

    st.markdown("### Filtres")
    f1, f2, f3, f4, f5 = st.columns(5)

    with f1:
        selected_periods = st.multiselect(
            "Période (mois / année)",
            options=sorted(docs_df["month"].dropna().unique().tolist()),
            default=[],
        )
    with f2:
        supplier_filter = st.multiselect(
            "Fournisseur",
            options=sorted(docs_df["supplier_name"].dropna().unique().tolist()),
            default=[],
        )
    with f3:
        category_filter = st.multiselect(
            "Catégorie",
            options=sorted(docs_df["category"].dropna().unique().tolist()),
            default=[],
        )
    with f4:
        doc_type_filter = st.multiselect(
            "Type document",
            options=sorted(docs_df["doc_type"].dropna().unique().tolist()),
            default=[],
        )
    with f5:
        all_amounts = docs_df["total_ttc"].fillna(0)
        min_default = float(all_amounts.min()) if len(all_amounts) else 0.0
        max_default = float(all_amounts.max()) if len(all_amounts) else 1000.0
        amount_range = st.slider(
            "Montant min / max",
            min_value=float(min_default),
            max_value=float(max_default if max_default > min_default else min_default + 1),
            value=(float(min_default), float(max_default if max_default > min_default else min_default + 1)),
        )

    extra1, extra2 = st.columns(2)
    with extra1:
        monthly_ca = st.number_input("CA total de la période filtrée (optionnel)", min_value=0.0, step=1000.0, value=0.0)
    with extra2:
        total_patients = st.number_input("Nombre de patients sur la période (optionnel)", min_value=0.0, step=1.0, value=0.0)

    view_docs, view_lines = prepare_filtered_data(
        docs_df, lines_df,
        supplier_filter, category_filter, doc_type_filter,
        selected_periods, amount_range[0], amount_range[1]
    )

    if view_docs.empty:
        st.warning("Aucune donnée après filtrage.")
        st.stop()

    supplier_metrics = compute_supplier_metrics(
        view_docs,
        monthly_ca=monthly_ca if monthly_ca > 0 else None,
        total_patients=total_patients if total_patients > 0 else None,
    )
    strategic_df = compute_supplier_strategic_table(view_docs)
    anomalies_df = build_audit_alerts(view_docs, view_lines)
    product_pack = compute_product_analytics(view_lines)
    auto_messages = generate_auto_analysis(view_docs, view_lines, supplier_metrics, strategic_df, anomalies_df)

    st.markdown("### KPIs clés")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Montant TTC", format_money(view_docs["total_ttc"].sum()))
    with c2:
        metric_card("Ticket moyen", format_money(supplier_metrics.get("ticket_moyen")))
    with c3:
        metric_card("Coût moyen / fournisseur", format_money(supplier_metrics.get("cout_moyen_fournisseur")))
    with c4:
        metric_card("Fréquence achat", f"{supplier_metrics.get('frequence_achat', 0):.1f}" if pd.notna(supplier_metrics.get("frequence_achat")) else "-", "nb factures / mois")
    with c5:
        metric_card("Taux concentration top 3", format_pct(supplier_metrics.get("top3_share")))
    with c6:
        metric_card("Top fournisseur à risque", f"{supplier_metrics.get('top_supplier') or '-'}", format_pct(supplier_metrics.get("top_supplier_share")))

    st.markdown("## Analyse financière fournisseurs")
    a1, a2 = st.columns(2)

    with a1:
        monthly_spend = supplier_metrics["monthly_spend"]
        if not monthly_spend.empty:
            fig = px.line(monthly_spend, x="month", y="total_ttc", markers=True, title="Evolution mensuelle des achats")
            st.plotly_chart(style_plot(fig), use_container_width=True)

    with a2:
        split_df = compute_charge_split(view_docs)
        fig = px.pie(split_df, names="type_charge", values="montant", hole=0.5, title="Répartition charges fixes vs variables")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        metric_card("Charges fournisseurs / CA", format_pct(supplier_metrics.get("supplier_ca_ratio")), "si CA saisi")
    with r2:
        medical_ratio = safe_div(
            view_docs.loc[view_docs["category"].fillna("").str.contains("medical|médic|analyses|sang|consommables", case=False, regex=True), "total_ttc"].sum(),
            view_docs["total_ttc"].sum()
        )
        metric_card("Charges médicales / total", format_pct(medical_ratio))
    with r3:
        metric_card("Coût par patient", format_money(supplier_metrics.get("cout_par_patient")), "si patients saisis")

    st.markdown("## Répartition analytique")
    b1, b2 = st.columns(2)

    with b1:
        category_df = compute_category_breakdown(view_docs)
        fig = px.bar(category_df, x="category", y="total_ttc", title="Dépenses par catégorie")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with b2:
        cost_center_df = compute_cost_center_breakdown(view_docs)
        fig = px.bar(cost_center_df, x="cost_center", y="total_ttc", title="Coût par centre de coût")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    heatmap_df = compute_heatmap_month_category(view_docs)
    if not heatmap_df.empty:
        st.markdown("### Heatmap dépenses par mois / catégorie")
        fig = px.imshow(
            heatmap_df,
            text_auto=".0f",
            aspect="auto",
            title="Heatmap dépenses par mois / catégorie",
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown("## Analyse stratégique fournisseurs")
    if strategic_df.empty:
        st.info("Pas assez de données pour l’analyse stratégique.")
    else:
        s1, s2 = st.columns(2)
        with s1:
            top_strat = strategic_df.sort_values("total_ttc", ascending=False).head(15)
            fig = px.bar(top_strat, x="supplier_name", y="total_ttc", color="abc_class", title="Classement ABC fournisseurs")
            st.plotly_chart(style_plot(fig), use_container_width=True)

        with s2:
            fig = px.scatter(
                strategic_df,
                x="nb_factures",
                y="total_ttc",
                size="supplier_score",
                color="abc_class",
                hover_data=["supplier_name", "dependance_pct", "price_stability_score"],
                title="Score fournisseur : volume, fréquence, stabilité prix"
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)

        show_strat = strategic_df.copy()
        show_strat["total_ttc"] = show_strat["total_ttc"].apply(format_money)
        show_strat["avg_ticket"] = show_strat["avg_ticket"].apply(format_money)
        show_strat["dependance_pct"] = show_strat["dependance_pct"].apply(format_pct)
        show_strat["supplier_score"] = show_strat["supplier_score"].round(1)
        show_strat["price_stability_score"] = show_strat["price_stability_score"].round(2)

        st.dataframe(
            show_strat[[
                "supplier_name", "abc_class", "total_ttc", "nb_factures", "avg_ticket",
                "dependance_pct", "price_stability_score", "supplier_score"
            ]],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("## Analyse des prix unitaires")
    product_summary = product_pack["product_summary"]
    monthly_price = product_pack["monthly_price"]
    supplier_price_compare = product_pack["supplier_price_compare"]
    inflation_table = product_pack["inflation_table"]

    if not view_lines.empty and not product_summary.empty:
        p1, p2 = st.columns(2)

        with p1:
            top_costly = product_summary.sort_values("cout_total", ascending=False).head(15)
            fig = px.bar(top_costly, x="product_name", y="cout_total", title="Top produits les plus coûteux")
            st.plotly_chart(style_plot(fig), use_container_width=True)

        with p2:
            top_qty = product_summary.sort_values("quantite_totale", ascending=False).head(15)
            fig = px.bar(top_qty, x="product_name", y="quantite_totale", title="Top produits les plus consommés")
            st.plotly_chart(style_plot(fig), use_container_width=True)

        product_summary_show = product_summary.copy()
        product_summary_show["cout_total"] = product_summary_show["cout_total"].apply(format_money)
        product_summary_show["cout_moyen"] = product_summary_show["cout_moyen"].apply(format_money)
        product_summary_show["prix_moyen"] = product_summary_show["prix_moyen"].apply(format_money)
        st.dataframe(
            product_summary_show.sort_values("quantite_totale", ascending=False).head(100),
            use_container_width=True,
            hide_index=True
        )

        selected_product = st.selectbox(
            "Choisir un produit pour suivre son prix",
            options=sorted(product_summary["product_name"].dropna().unique().tolist())
        )

        pr1, pr2 = st.columns(2)
        with pr1:
            product_month_curve = monthly_price[monthly_price["product_name"] == selected_product].sort_values("month")
            if not product_month_curve.empty:
                fig = px.line(product_month_curve, x="month", y="unit_price", markers=True, title=f"Evolution du prix - {selected_product}")
                st.plotly_chart(style_plot(fig), use_container_width=True)

        with pr2:
            product_compare = supplier_price_compare[supplier_price_compare["product_name"] == selected_product].sort_values("unit_price", ascending=False)
            if not product_compare.empty:
                fig = px.bar(product_compare, x="supplier_name", y="unit_price", title=f"Comparaison prix entre fournisseurs - {selected_product}")
                st.plotly_chart(style_plot(fig), use_container_width=True)

        surcout_df = supplier_price_compare.copy()
        if not surcout_df.empty:
            min_price_by_product = surcout_df.groupby("product_name")["unit_price"].min().reset_index().rename(columns={"unit_price": "min_price"})
            surcout_df = surcout_df.merge(min_price_by_product, on="product_name", how="left")
            surcout_df["surcout"] = surcout_df["unit_price"] - surcout_df["min_price"]
            surcout_df["surcout_pct"] = np.where(
                surcout_df["min_price"].fillna(0) > 0,
                surcout_df["surcout"] / surcout_df["min_price"],
                np.nan
            )
            surcout_show = surcout_df[surcout_df["surcout"] > 0].sort_values("surcout_pct", ascending=False).head(50).copy()
            if not surcout_show.empty:
                st.markdown("### Détection surcoût")
                surcout_show["unit_price"] = surcout_show["unit_price"].apply(format_money)
                surcout_show["min_price"] = surcout_show["min_price"].apply(format_money)
                surcout_show["surcout"] = surcout_show["surcout"].apply(format_money)
                surcout_show["surcout_pct"] = surcout_show["surcout_pct"].apply(format_pct)
                st.dataframe(
                    surcout_show[["product_name", "supplier_name", "unit_price", "min_price", "surcout", "surcout_pct"]],
                    use_container_width=True,
                    hide_index=True
                )

            inflation_show = inflation_table[
                inflation_table["variation_pct"].fillna(0) > 0.10
            ].sort_values("variation_pct", ascending=False).head(50).copy()
            if not inflation_show.empty:
                st.markdown("### Détection inflation cachée")
                inflation_show["unit_price"] = inflation_show["unit_price"].apply(format_money)
                inflation_show["previous_price"] = inflation_show["previous_price"].apply(format_money)
                inflation_show["variation_pct"] = inflation_show["variation_pct"].apply(format_pct)
                st.dataframe(
                    inflation_show[["month", "product_name", "previous_price", "unit_price", "variation_pct"]],
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.info("Pas assez de lignes détaillées pour l’analyse des prix unitaires.")

    st.markdown("## Analyse automatique (IA)")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    for msg in auto_messages:
        st.write(f"• {msg}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## Export")
    excel_bytes = build_excel_report(view_docs, view_lines, anomalies_df, strategic_df, product_pack)
    st.download_button(
        "Télécharger l’export Excel retraité",
        data=excel_bytes,
        file_name="rapport_analyse_fournisseurs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    pdf_bytes = build_pdf_report(view_docs, anomalies_df, auto_messages)
    if pdf_bytes is not None:
        st.download_button(
            "Télécharger le rapport PDF audit",
            data=pdf_bytes,
            file_name="rapport_audit_fournisseurs.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Le PDF nécessite le package reportlab. Si le bouton ne s’affiche pas, ajoute reportlab dans requirements.txt.")

# =========================================================
# PAGE AUDIT
# =========================================================
elif menu == "Audit":
    st.subheader("Audit fournisseurs et contrôle des anomalies")

    docs_df, lines_df, mappings_df, suppliers_df = load_all_data()
    audit_df = build_audit_alerts(docs_df, lines_df)

    if docs_df.empty:
        st.info("Aucun document enregistré.")
        st.stop()

    total_alerts = len(audit_df)
    high_risk = int((audit_df["niveau_risque"] == "Élevé").sum()) if not audit_df.empty else 0
    med_risk = int((audit_df["niveau_risque"] == "Moyen").sum()) if not audit_df.empty else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Alertes", str(total_alerts))
    with c2:
        metric_card("Risque élevé", str(high_risk))
    with c3:
        metric_card("Risque moyen", str(med_risk))

    st.markdown("### Badges de risque")
    badge_parts = []
    if high_risk > 0:
        badge_parts.append(badge_html(f"{high_risk} alertes élevées", "red"))
    if med_risk > 0:
        badge_parts.append(badge_html(f"{med_risk} alertes moyennes", "orange"))
    if total_alerts == 0:
        badge_parts.append(badge_html("Aucune anomalie détectée", "green"))
    st.markdown("".join(badge_parts), unsafe_allow_html=True)

    if audit_df.empty:
        st.success("Aucune alerte détectée sur les règles actuelles.")
        st.stop()

    st.markdown("### Tableau anomalies")
    show = audit_df.copy()
    show["valeur"] = show["valeur"].apply(format_money)
    st.dataframe(
        show[["type_controle", "niveau_risque", "supplier_name", "reference", "observation", "valeur"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Répartition des anomalies")
    ar1, ar2 = st.columns(2)
    with ar1:
        by_type = audit_df.groupby("type_controle").size().reset_index(name="nb").sort_values("nb", ascending=False)
        fig = px.bar(by_type, x="type_controle", y="nb", title="Anomalies par type")
        st.plotly_chart(style_plot(fig), use_container_width=True)
    with ar2:
        by_risk = audit_df.groupby("niveau_risque").size().reset_index(name="nb")
        fig = px.pie(by_risk, names="niveau_risque", values="nb", title="Anomalies par niveau de risque", hole=0.5)
        st.plotly_chart(style_plot(fig), use_container_width=True)

    auto_messages = generate_auto_analysis(docs_df, lines_df, compute_supplier_metrics(docs_df), compute_supplier_strategic_table(docs_df), audit_df)

    pdf_bytes = build_pdf_report(docs_df, audit_df, auto_messages)
    if pdf_bytes is not None:
        st.download_button(
            "Exporter PDF rapport audit",
            data=pdf_bytes,
            file_name="rapport_audit_fournisseurs.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Le PDF nécessite reportlab dans requirements.txt.")

# =========================================================
# PAGE MAPPINGS
# =========================================================
elif menu == "Mappings analytiques":
    st.subheader("Mappings analytiques")
    docs_df, lines_df, mappings_df, suppliers_df = load_all_data()
    st.dataframe(mappings_df, use_container_width=True, hide_index=True)

# =========================================================
# PAGE FOURNISSEURS
# =========================================================
elif menu == "Référentiel fournisseurs":
    st.subheader("Référentiel fournisseurs")
    docs_df, lines_df, mappings_df, suppliers_df = load_all_data()

    if suppliers_df.empty:
        st.info("Aucun fournisseur enregistré pour le moment.")
    else:
        st.dataframe(suppliers_df, use_container_width=True, hide_index=True)