import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Drift Lens", layout="wide")

RAW_URL = "https://raw.githubusercontent.com/mcarolina2/drift_lens/main/drift_report.json"

@st.cache_data(ttl=300)  # atualiza a cada 5 min
def carregar_dados():
    r = requests.get(RAW_URL)
    r.raise_for_status()
    return r.json()

st.title("Drift Lens — Monitoramento de Data Drift")

try:
    dados = carregar_dados()
except Exception as e:
    st.error(f"Não foi possível carregar o relatório: {e}")
    st.stop()

# Ajuste as chaves conforme a estrutura real do seu drift_report.json
df = pd.DataFrame(dados["features"])  # ex: [{"feature": "renda", "psi": 0.31, "status": "CRÍTICO"}, ...]

col1, col2, col3 = st.columns(3)
col1.metric("Features monitoradas", len(df))
col2.metric("Em drift crítico", (df["status"] == "CRÍTICO").sum())
col3.metric("Estáveis", (df["status"] == "ESTÁVEL").sum())

st.subheader("Detalhamento por feature")
st.dataframe(df, use_container_width=True)

st.subheader("PSI por feature")
st.bar_chart(df.set_index("feature")["psi"])

st.caption(f"Última atualização: {dados.get('timestamp', 'desconhecida')}")