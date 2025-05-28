import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import os
from dotenv import load_dotenv
from fpdf import FPDF
import datetime

def get_esg_news_newsdata(supplier_id):
    import os, requests
    api_key = os.getenv("NEWSDATA_API_KEY")
    url = "https://newsdata.io/api/1/news"
    query = f'"{supplier_id}" AND (ESG OR sustainability OR sanction OR "forced labor" OR corruption)'
    params = {
        "apikey": api_key,
        "q": query,
        "language": "en",
        "category": "business,politics"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        st.error(f"News API error: {e}")
        return []

def check_sanctions(supplier_id):
    import os, requests
    api_key = os.getenv("OPENSANCTIONS_API_KEY")
    url = "https://api.opensanctions.org/v1/entities/_search"
    params = {
        "q": supplier_id,
        "dataset": "sanctions",
        "api_key": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            return True, [result["name"] for result in data["results"]]
        return False, []
    except Exception as e:
        st.error(f"Sanctions check failed: {e}")
        return False, []

# --- Load environment variables ---
load_dotenv()
SONAR_API_KEY = os.getenv("SONAR_API_KEY")
SONAR_API_URL = os.getenv("SONAR_API_URL")

st.set_page_config(page_title="Ethical Supply Chain Mapper (ESC-M)", layout="wide")
st.title("🌍 Ethical Supply Chain Mapper (ESC-M)")

# --- Utility Functions ---

def get_esg_risk_report(prompt, model="sonar"):
    headers = {
        "Authorization": f"Bearer {SONAR_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an ESG compliance expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(SONAR_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "No response from Sonar API.")

def score_esg_risk(row):
    # Simple scoring: sum of all principle columns (customize as needed)
    cols = [col for col in row.index if "PRINCIPLE" in col.upper()]
    score = sum([row[c] for c in cols])
    if score < 10:
        return "low_risk"
    elif score < 20:
        return "moderate_risk"
    elif score < 30:
        return "high_risk"
    else:
        return "dangerous_risk"

def build_multi_tier_graph(df, tiers=3):
    # Simulate multi-tier mapping based on id for demo purposes
    G = nx.DiGraph()
    ids = df['id'].astype(str).tolist()
    for i, row in df.iterrows():
        supplier = f"Supplier_{row['id']}"
        G.add_node(supplier, tier=1, risk=row['Predicted_Risk'])
        # Add tier 2 and 3 suppliers (simulate relationships)
        if i+1 < len(ids):
            t2 = f"Supplier_{ids[i+1]}"
            G.add_edge(supplier, t2, tier=2, risk=row['Predicted_Risk'])
            if i+2 < len(ids):
                t3 = f"Supplier_{ids[i+2]}"
                G.add_edge(t2, t3, tier=3, risk=row['Predicted_Risk'])
    return G

def draw_graph(G):
    pos = nx.spring_layout(G, seed=42)
    risks = nx.get_edge_attributes(G, 'risk')
    colors = []
    for u, v in G.edges():
        risk = risks.get((u, v), 'unknown')
        if risk == 'high_risk':
            colors.append('red')
        elif risk == 'moderate_risk':
            colors.append('orange')
        elif risk == 'low_risk':
            colors.append('green')
        else:
            colors.append('gray')
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, edge_color=colors, node_color='skyblue', font_size=10, arrows=True)
    plt.title("Multi-Tier Supply Chain Knowledge Graph")
    st.pyplot(plt)

from fpdf import FPDF

def generate_compliance_report_pdf(df, ai_reports):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="ESC-M Compliance Report", ln=True, align='C')
    pdf.ln(10)
    for idx, row in df.iterrows():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Supplier: {row['id']}", ln=True)
        pdf.set_font("Arial", "", 11)
        report = ai_reports.get(str(row['id']), 'No AI report available.')
        pdf.multi_cell(0, 6, report)
        pdf.ln(2)
    return pdf.output(dest='S').encode('latin1')


def simulate_real_time_alerts(df):
    # Simulate a real-time alert for any high/dangerous risk supplier
    alerts = []
    for idx, row in df.iterrows():
        if row['Predicted_Risk'] in ['high_risk', 'dangerous_risk']:
            alerts.append(f"🚨 Real-time alert: Supplier ID {row['id']} flagged as {row['Predicted_Risk'].replace('_', ' ').title()}!")
    return alerts

def recommend_alternative_supplier(df, selected_row):
    # Recommend a supplier with lowest risk and similar scores
    safe_suppliers = df[df['Predicted_Risk'] == 'low_risk']
    if safe_suppliers.empty:
        return "No low-risk alternative suppliers found."
    # Pick the one with closest 'COUNTRY_SECTOR_AVERAGE'
    safe_suppliers['diff'] = (safe_suppliers['COUNTRY_SECTOR_AVERAGE'] - selected_row['COUNTRY_SECTOR_AVERAGE']).abs()
    best = safe_suppliers.sort_values('diff').iloc[0]
    return f"Recommended alternative: Supplier ID {best['id']} (Risk: {best['Predicted_Risk']})"

# --- ESC-M Workflow Tabs ---

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Data & Risk Mapping",
    "2️⃣ Multi-Tier Graph",
    "3️⃣ Real-Time Alerts",
    "4️⃣ Insights & Actions",
    "5️⃣ Compliance Reports"
])

with tab1:
    st.header("Step 1: Supplier Data Integration & ESG Risk Mapping")
    uploaded_file = st.file_uploader("Upload your supplier ESG data CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, delimiter=';', skiprows=2, header=0)
        df = df.dropna(axis=1, how='all')
        df.columns = df.columns.str.strip()
        st.success("Supplier data loaded!")
        st.dataframe(df.head())

        # Risk scoring
        df['Predicted_Risk'] = df.apply(score_esg_risk, axis=1)
        st.subheader("Predicted ESG Risk Levels")
        st.dataframe(df[['id', 'Predicted_Risk']])

        # Save to session for other tabs
        st.session_state['escm_df'] = df

with tab2:
    st.header("Step 2: Multi-Tier Supply Chain Knowledge Graph")
    df = st.session_state.get('escm_df')
    if df is not None:
        G = build_multi_tier_graph(df)
        draw_graph(G)

        # Risk distribution chart
        st.subheader("Risk Level Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df['Predicted_Risk'], order=sorted(df['Predicted_Risk'].unique()), ax=ax)
        st.pyplot(fig)
    else:
        st.info("Please upload supplier data in Step 1.")

with tab3:
    st.header("🚨 Real-Time ESG & Sanctions Alerts")
    df = st.session_state.get('escm_df')  # or however you store your DataFrame
    if df is not None:
        for _, row in df.iterrows():
            supplier_id = str(row["id"])
            # Sanctions Check
            flagged, matches = check_sanctions(supplier_id)
            if flagged:
                st.error(f"⚠️ Sanction Alert: Supplier ID `{supplier_id}` matches: {', '.join(matches)}")
            # ESG News Check
            news = get_esg_news_newsdata(supplier_id)
            if news:
                for article in news[:2]:
                    st.warning(f"📰 ESG News for `{supplier_id}`: [{article['title']}]({article['link']}) ({article['pubDate']})")
            if not flagged and not news:
                st.info(f"No alerts for `{supplier_id}`.")
    else:
        st.info("Upload or process supplier data in Step 1 first.")

with tab4:
    st.header("Step 4: Insights & AI Actions")
    df = st.session_state.get('escm_df')  # Adjust this if your DataFrame is named differently
    if df is not None and not df.empty:
        # Use the 'id' column for the dropdown, since it contains company names
        company_names = df['id'].astype(str).unique().tolist()
        selected_company = st.selectbox("Select a company:", company_names)
        
        # Get the row for the selected company
        selected_row = df[df['id'].astype(str) == selected_company].iloc[0]
        
        # Show company info
        st.write("**Company details:**")
        st.write(selected_row)
        
        # Build prompt for AI report (adapt as needed)
        esg_prompt = f"""
        Company Name: {selected_row['id']}
        Risk Level: {selected_row.get('Predicted_Risk', 'NA')}
        Trend RRI: {selected_row.get('TREND_RRI', 'NA')}
        Current RRI: {selected_row.get('CURRENT_RRI', 'NA')}
        Country Avg: {selected_row.get('COUNTRY_SECTOR_AVERAGE', 'NA')}
        Compliance Scores:
        - Anti-Corruption: {selected_row.get('PRINCIPLE_10_ANTI_CORRUPTION', 'NA')}
        - Human Rights: {selected_row.get('PRINCIPLE_1_HUMAN_RIGHTS', 'NA')}
        - Labour Standards: {selected_row.get('PRINCIPLE_3_LABOUR', 'NA')}
        - Environmental: {selected_row.get('PRINCIPLE_7_ENVIRONMENT', 'NA')}
        Provide:
        1. Risk factors analysis
        2. 3 improvement recommendations
        3. Priority actions
        """
        if st.button("Generate AI ESG Risk Report"):
            try:
                ai_report = get_esg_risk_report(esg_prompt)
                st.subheader(f"📄 AI ESG Risk Report for {selected_row['id']}")
                st.write(ai_report)
            except Exception as e:
                st.error(f"AI analysis failed: {e}")
    else:
        st.info("Please upload and process your data first.")


with tab5:
    st.header("Step 5: Compliance Reports")
    df = st.session_state.get('escm_df')  # Your main DataFrame
    ai_reports = st.session_state.get('ai_reports', {})
    if df is not None and ai_reports:
        pdf_bytes = generate_compliance_report_pdf(df, ai_reports)
        st.download_button(
            label="Download Compliance Report (PDF)",
            data=pdf_bytes,
            file_name="escm_compliance_report.pdf",
            mime="application/pdf"
        )
        st.download_button(
            label="Download Raw Data (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="escm_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Generate at least one AI report to enable compliance report download.")


st.markdown("---")
st.caption("ESC-M: Ethical Supply Chain Mapper | AI-powered supply chain ESG risk intelligence")
