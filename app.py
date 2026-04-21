import streamlit as st
import pandas as pd

st.set_page_config(page_title="Analyse Factures", layout="wide")

st.title("📊 Analyse des factures fournisseurs")

uploaded_file = st.file_uploader("📂 Importer fichier Excel", type=["xls", "xlsx"])

if uploaded_file:

    # --------------------------------------------------
    # 📥 LECTURE FICHIER BRUT
    # --------------------------------------------------
    df = pd.read_excel(uploaded_file, header=None)

    st.subheader("🔍 Aperçu fichier brut")
    st.dataframe(df.head(30))

    # --------------------------------------------------
    # 🧠 PARSING INTELLIGENT
    # --------------------------------------------------
    results = []
    current_invoice = None

    for i in range(len(df)):
        row = df.iloc[i]

        # 🔵 LIGNE FACTURE
        if pd.notna(row[0]) and pd.notna(row[2]) and pd.notna(row[4]):
            try:
                current_invoice = {
                    "supplier_name": str(row[1]).strip(),
                    "invoice_number": str(row[2]).strip(),
                    "invoice_date": pd.to_datetime(row[3], errors='coerce'),
                    "total_ttc": float(row[4]) if pd.notna(row[4]) else 0,
                    "paid_amount": 0,
                    "payment_date": None
                }
                results.append(current_invoice)
            except:
                continue

        # ⚪ LIGNE PAIEMENT
        elif current_invoice is not None:
            if str(row[2]).lower() in ["virement", "espèce", "cheque", "chèque"]:
                try:
                    payment_date = pd.to_datetime(row[1], errors='coerce')
                    amount = float(row[4]) if pd.notna(row[4]) else 0

                    current_invoice["paid_amount"] += amount
                    current_invoice["payment_date"] = payment_date
                except:
                    pass

    # --------------------------------------------------
    # 📊 DATAFRAME FINAL
    # --------------------------------------------------
    result_df = pd.DataFrame(results)

    if result_df.empty:
        st.warning("❌ Aucune facture détectée")
        st.stop()

    # nettoyage dates
    result_df["invoice_date"] = pd.to_datetime(result_df["invoice_date"], errors='coerce')
    result_df["payment_date"] = pd.to_datetime(result_df["payment_date"], errors='coerce')

    # --------------------------------------------------
    # 📊 STATUT PAIEMENT
    # --------------------------------------------------
    result_df["status"] = result_df.apply(
        lambda x: "Payée" if x["paid_amount"] >= x["total_ttc"]
        else ("Partielle" if x["paid_amount"] > 0 else "Non payée"),
        axis=1
    )

    st.subheader("✅ Factures détectées")
    st.dataframe(result_df)

    # --------------------------------------------------
    # ⏱️ CALCUL DELAIS
    # --------------------------------------------------
    result_df["payment_delay_days"] = result_df.apply(
        lambda x: (pd.Timestamp.today() - x["invoice_date"]).days
        if pd.isna(x["payment_date"])
        else (x["payment_date"] - x["invoice_date"]).days,
        axis=1
    )

    # --------------------------------------------------
    # ⚙️ PARAMETRAGE DELAIS FOURNISSEURS
    # --------------------------------------------------
    st.subheader("⚙️ Délais de paiement par fournisseur")

    suppliers = result_df["supplier_name"].dropna().unique()
    supplier_terms = {}

    for supplier in suppliers:
        supplier_terms[supplier] = st.number_input(
            f"{supplier}",
            min_value=0,
            max_value=365,
            value=30,
            key=supplier
        )

    result_df["expected_delay"] = result_df["supplier_name"].map(supplier_terms)

    # --------------------------------------------------
    # 🚨 ANALYSE RETARD
    # --------------------------------------------------
    result_df["delay_status"] = result_df.apply(
        lambda x: "Respecté"
        if x["payment_delay_days"] <= x["expected_delay"]
        else "En retard",
        axis=1
    )

    result_df["delay_over"] = result_df.apply(
        lambda x: max(0, x["payment_delay_days"] - x["expected_delay"]),
        axis=1
    )

    st.subheader("📋 Analyse des délais")
    st.dataframe(result_df)

    # --------------------------------------------------
    # 📈 KPI GLOBAL
    # --------------------------------------------------
    st.subheader("📈 Indicateurs globaux")

    total = len(result_df)
    on_time = len(result_df[result_df["delay_status"] == "Respecté"])
    late = len(result_df[result_df["delay_status"] == "En retard"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total factures", total)
    col2.metric("Respectées", on_time)
    col3.metric("En retard", late)

    # --------------------------------------------------
    # 📊 ANALYSE PAR FOURNISSEUR
    # --------------------------------------------------
    st.subheader("📊 Analyse par fournisseur")

    supplier_analysis = result_df.groupby("supplier_name").agg(
        total_factures=("invoice_number", "count"),
        delai_moyen=("payment_delay_days", "mean"),
        retard_moyen=("delay_over", "mean"),
        respect_count=("delay_status", lambda x: (x == "Respecté").sum())
    ).reset_index()

    supplier_analysis["taux_respect_%"] = (
        supplier_analysis["respect_count"] / supplier_analysis["total_factures"]
    ) * 100

    st.dataframe(supplier_analysis)