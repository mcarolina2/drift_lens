import sys
import os
import streamlit as st
import pandas as pd

# permite importar src/github_api.py a partir da pasta dashboard/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from github_api import get_drift_report_com_metadados

st.set_page_config(page_title="Drift Lens", layout="wide")

OWNER = "mcarolina2"   # substitua pelo seu usuário/organização real
REPO = "drift_lens"
BRANCH = "main"


@st.cache_data(ttl=300)
def carregar_dados():
    return get_drift_report_com_metadados(OWNER, REPO, "drift_report.json", BRANCH)


st.title("Drift Lens — Monitoramento de Data Drift")

try:
    dados = carregar_dados()
except Exception as e:
    st.error(f"Não foi possível carregar o relatório: {e}")
    st.stop()

df = pd.DataFrame(dados["features"])

col1, col2, col3 = st.columns(3)
col1.metric("Features monitoradas", len(df))
col2.metric("Em drift crítico", (df["status"] == "CRÍTICO").sum())
col3.metric("Estáveis", (df["status"] == "ESTÁVEL").sum())

st.subheader("Detalhamento por feature")
st.dataframe(df, use_container_width=True)

st.subheader("PSI por feature")
st.bar_chart(df.set_index("feature")["psi"])

# ── Informações do commit que gerou o relatório ──────────────────
commit = dados.get("_commit", {})
with st.expander("Origem deste relatório"):
    st.write(f"**Autor:** {commit.get('autor', '—')}")
    st.write(f"**Data do commit:** {commit.get('data', '—')}")
    st.write(f"**Mensagem:** {commit.get('mensagem', '—')}")
    if commit.get("url"):
        st.markdown(f"[Ver commit no GitHub]({commit['url']})")

st.caption(f"Relatório gerado em: {dados.get('timestamp', 'desconhecida')}")