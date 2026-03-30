import os
import re
import hmac
import uuid
import time
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
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
        max-width: 1500px;
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

    # =========================
    # Corrections en-tête
    # =========================
    if "agence marocaine du sang" in raw_norm or "agence marocaine du sang" in file_norm:
        vendor_name = "Agence Marocaine du Sang et de ses Dérivés"

    if not vendor_name and "agence marocaine du sang" in raw_norm:
        vendor_name = "Agence Marocaine du Sang et de ses Dérivés"

    # Numéro facture : garder juste le bon format YYYY/NNNNNN
    if invoice_id:
        m = re.search(r"(\d{4}/\d{4,})", str(invoice_id))
        if m:
            invoice_id = m.group(1)
    else:
        m = re.search(r"(\d{4}/\d{4,})", raw_text)
        if m:
            invoice_id = m.group(1)

    # Client
    if not customer_name:
        m = re.search(r"Client\s*:\s*(.+)", raw_text, flags=re.IGNORECASE)
        if m:
            customer_name = m.group(1).strip()

    # Période du ... au ...
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

    # Date facture = date début si Azure ne la donne pas
    if not invoice_date and period_start:
        invoice_date = period_start

    # Devise
    if "dirham" in raw_norm or "mad" in raw_norm or "sept mille deux cent dirhams" in raw_norm:
        currency = "MAD"
    if "agence marocaine du sang" in raw_norm:
        currency = "MAD"

    # Total TTC fallback
    if pd.isna(amount_due):
        m = re.search(r"M\.?TOTAL\s*([0-9]+[.,]?[0-9]*)", raw_text, flags=re.IGNORECASE)
        if m:
            amount_due = parse_numeric_value(m.group(1))

    # =========================
    # Lignes Azure
    # =========================
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

    # =========================
    # Post-traitement spécial Agence du Sang
    # =========================
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
# EXTRACTION ENTRYPOINT
# =========================================================
def azure_is_configured():
    return bool(AZURE_DI_ENDPOINT and AZURE_DI_KEY)

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

        # Format attendu :
        # BELYCH MOSTAPHA Culot Globulaire (unité) 1483326 18/03/2025 2 360.00 720.00
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

    if not lines.empty:
        for c in ["service_date"]:
            if c in lines.columns:
                lines[c] = pd.to_datetime(lines[c], errors="coerce")
        for c in ["quantity", "unit_price_ht", "unit_price_ttc", "discount", "vat_rate", "line_amount_ht", "line_amount_ttc"]:
            if c in lines.columns:
                lines[c] = pd.to_numeric(lines[c], errors="coerce")

    return docs, lines, mappings, suppliers


# =========================================================
# KPI / AUDIT
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


def build_audit_alerts(docs_df, lines_df):
    alerts = []

    if docs_df.empty:
        return pd.DataFrame(alerts)

    if "invoice_number" in docs_df.columns:
        dup = docs_df[
            docs_df["invoice_number"].fillna("").astype(str).str.strip().ne("")
            & docs_df["invoice_number"].fillna("").duplicated(keep=False)
        ]
        for _, row in dup.iterrows():
            alerts.append({
                "type_controle": "Doublon numéro facture",
                "niveau_risque": "Élevé",
                "supplier_name": row.get("supplier_name"),
                "reference": row.get("invoice_number"),
                "observation": "Même numéro détecté plusieurs fois",
                "valeur": row.get("total_ttc"),
            })

    zero_total = docs_df[pd.to_numeric(docs_df["total_ttc"], errors="coerce").fillna(0) == 0]
    for _, row in zero_total.iterrows():
        alerts.append({
            "type_controle": "Montant nul ou absent",
            "niveau_risque": "Élevé",
            "supplier_name": row.get("supplier_name"),
            "reference": row.get("invoice_number"),
            "observation": "Total TTC nul ou non extrait",
            "valeur": row.get("total_ttc"),
        })

    miss_date = docs_df[docs_df["document_date"].isna()]
    for _, row in miss_date.iterrows():
        alerts.append({
            "type_controle": "Date absente",
            "niveau_risque": "Moyen",
            "supplier_name": row.get("supplier_name"),
            "reference": row.get("invoice_number"),
            "observation": "Date document non détectée",
            "valeur": row.get("total_ttc"),
        })

    if not lines_df.empty:
        check = lines_df.copy()
        check["calc_amount"] = check["quantity"] * check["unit_price_ttc"]
        check["delta"] = (check["line_amount_ttc"] - check["calc_amount"]).abs()
        incoherent = check[
            pd.notna(check["quantity"])
            & pd.notna(check["unit_price_ttc"])
            & pd.notna(check["line_amount_ttc"])
            & (check["delta"] > 5)
        ]

        for _, row in incoherent.head(300).iterrows():
            alerts.append({
                "type_controle": "Écart ligne / PU x Qté",
                "niveau_risque": "Moyen",
                "supplier_name": None,
                "reference": row.get("designation"),
                "observation": "Montant ligne différent du calcul Qté x PU",
                "valeur": row.get("line_amount_ttc"),
            })

    return pd.DataFrame(alerts)


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
        "Base Lignes",
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
    st.write("Extraction robuste via Azure Document Intelligence + base documentaire + analyse.")

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
# PAGE BASE LIGNES
# =========================================================
elif menu == "Base Lignes":
    st.subheader("Base lignes")

    docs_df, lines_df, mappings_df, suppliers_df = load_all_data()

    if lines_df.empty:
        st.info("Aucune ligne enregistrée.")
        st.stop()

    show = lines_df.copy()
    for c in ["unit_price_ht", "unit_price_ttc", "line_amount_ht", "line_amount_ttc"]:
        if c in show.columns:
            show[c] = show[c].apply(format_money)

    cols = [
        "document_id", "source_file", "line_no", "reference", "designation",
        "lot_number", "service_date", "patient_name", "quantity",
        "unit_price_ht", "unit_price_ttc", "discount", "vat_rate",
        "line_amount_ht", "line_amount_ttc", "cost_center", "charge_category"
    ]
    cols = [c for c in cols if c in show.columns]

    st.dataframe(show[cols], use_container_width=True, hide_index=True)

# =========================================================
# PAGE ANALYSE
# =========================================================
elif menu == "Analyse":
    st.subheader("Analyse analytique et fournisseurs")

    docs_df, lines_df, mappings_df, suppliers_df = load_all_data()

    if docs_df.empty:
        st.info("Aucun document enregistré.")
        st.stop()

    filt1, filt2, filt3 = st.columns(3)

    with filt1:
        supplier_filter = st.multiselect(
            "Fournisseurs",
            sorted([x for x in docs_df["supplier_name"].dropna().unique().tolist()]),
        )
    with filt2:
        category_filter = st.multiselect(
            "Catégories",
            sorted([x for x in docs_df["category"].dropna().unique().tolist()]),
        )
    with filt3:
        doc_type_filter = st.multiselect(
            "Types document",
            sorted([x for x in docs_df["doc_type"].dropna().unique().tolist()]),
        )

    view_docs = docs_df.copy()
    if supplier_filter:
        view_docs = view_docs[view_docs["supplier_name"].isin(supplier_filter)]
    if category_filter:
        view_docs = view_docs[view_docs["category"].isin(category_filter)]
    if doc_type_filter:
        view_docs = view_docs[view_docs["doc_type"].isin(doc_type_filter)]

    if view_docs.empty:
        st.warning("Aucune donnée après filtrage.")
        st.stop()

    joined_lines = (
        lines_df.merge(view_docs[["document_id"]], on="document_id", how="inner")
        if not lines_df.empty else pd.DataFrame()
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Montant TTC", format_money(view_docs["total_ttc"].sum()))
    with c2:
        metric_card("Nb documents", str(len(view_docs)))
    with c3:
        metric_card("Nb fournisseurs", str(view_docs["supplier_name"].nunique()))
    with c4:
        metric_card("Montant moyen", format_money(view_docs["total_ttc"].mean()))

    g1, g2 = st.columns(2)

    with g1:
        by_supplier = (
            view_docs.groupby("supplier_name", dropna=False)["total_ttc"]
            .sum()
            .reset_index()
            .sort_values("total_ttc", ascending=False)
            .head(15)
        )
        fig = px.bar(by_supplier, x="supplier_name", y="total_ttc", title="Top fournisseurs par montant TTC")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with g2:
        by_category = (
            view_docs.groupby("category", dropna=False)["total_ttc"]
            .sum()
            .reset_index()
            .sort_values("total_ttc", ascending=False)
        )
        fig = px.pie(by_category, names="category", values="total_ttc", hole=0.5, title="Répartition par catégorie")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    if not joined_lines.empty:
        st.markdown("### Analyse lignes")
        top_design = (
            joined_lines.groupby("designation", dropna=False)["line_amount_ttc"]
            .sum()
            .reset_index()
            .sort_values("line_amount_ttc", ascending=False)
            .head(20)
        )
        fig = px.bar(top_design, x="designation", y="line_amount_ttc", title="Top désignations par montant")
        st.plotly_chart(style_plot(fig), use_container_width=True)

# =========================================================
# PAGE AUDIT
# =========================================================
elif menu == "Audit":
    st.subheader("Audit des documents et lignes")

    docs_df, lines_df, mappings_df, suppliers_df = load_all_data()
    audit_df = build_audit_alerts(docs_df, lines_df)

    if docs_df.empty:
        st.info("Aucun document enregistré.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Alertes", str(len(audit_df)))
    with c2:
        metric_card("Risque élevé", str(int((audit_df["niveau_risque"] == "Élevé").sum()) if not audit_df.empty else 0))
    with c3:
        metric_card("Risque moyen", str(int((audit_df["niveau_risque"] == "Moyen").sum()) if not audit_df.empty else 0))

    if audit_df.empty:
        st.success("Aucune alerte détectée sur les règles actuelles.")
    else:
        show = audit_df.copy()
        show["valeur"] = show["valeur"].apply(format_money)
        st.dataframe(show, use_container_width=True, hide_index=True)

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