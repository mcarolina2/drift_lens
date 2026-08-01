"""
dashboard/app.py
Dashboard de monitoramento de Data Drift — Drift Lens.

Fontes de dados (em ordem de prioridade):
    1. GitHub API  — histórico de runs via artefatos do GitHub Actions
    2. Local       — drift_report.json gerado localmente

Configuração:
    Crie um arquivo .env na raiz com:
        GITHUB_TOKEN=ghp_...
        REPO_OWNER=mcarolina2
        REPO_NAME=drift-lens

Rodar:
    streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.github_api import load_drift_data, drift_probability_score, retrain_recommendation


# ─────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Drift Lens · Monitor de Data Drift",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.source-badge {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
}
.badge-github { background: #1f883d22; color: #1f883d; }
.badge-local  { background: #bf8e0022; color: #bf8e00; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Drift Lens")
    st.caption("Monitor de Data Drift para MLOps")
    st.divider()

    st.markdown("**Thresholds de alerta**")
    psi_warn = st.slider("PSI — moderado",  0.05, 0.20, 0.10, 0.01)
    psi_crit = st.slider("PSI — crítico",   0.10, 0.50, 0.20, 0.01)
    ks_alpha = st.slider("KS — alfa (p)",   0.01, 0.10, 0.05, 0.01)

    st.divider()
    st.markdown("**Pesos do Drift Score**")
    w_psi = st.slider("Peso PSI", 0.0, 1.0, 0.5, 0.1)
    w_ks  = st.slider("Peso KS",  0.0, 1.0, 0.3, 0.1)
    w_kl  = st.slider("Peso KL",  0.0, 1.0, 0.2, 0.1)

    total = w_psi + w_ks + w_kl
    if total > 0:
        w_psi, w_ks, w_kl = w_psi/total, w_ks/total, w_kl/total
    st.caption(f"Normalizados: PSI={w_psi:.2f} KS={w_ks:.2f} KL={w_kl:.2f}")

    st.divider()
    if st.button("🔄 Atualizar dados"):
        st.cache_data.clear()


# ─────────────────────────────────────────
# Carregamento de dados
# ─────────────────────────────────────────

@st.cache_data(ttl=300)
def get_data(w_psi, w_ks, w_kl):
    return load_drift_data(w_psi, w_ks, w_kl)


with st.spinner("Carregando dados..."):
    try:
        df, fonte = get_data(w_psi, w_ks, w_kl)
    except FileNotFoundError:
        st.error(
            "Nenhum dado encontrado. Execute primeiro:\n\n"
            "```bash\n"
            "python src/generate_test_data.py\n"
            "python src/drift_metrics.py "
            "--reference data/reference/train_data.csv "
            "--production data/production/serving_data.csv "
            "--output drift_report.json\n"
            "```"
        )
        st.stop()

if df.empty:
    st.warning("Nenhum artefato de drift encontrado no repositório.")
    st.stop()

# Recalcula scores com pesos atuais do sidebar
df["drift_score"] = df.apply(
    lambda r: drift_probability_score(r["psi"], r["ks_p_value"], r["kl_divergence"], w_psi, w_ks, w_kl),
    axis=1
)
df["recommendation"] = df["drift_score"].apply(lambda s: retrain_recommendation(s)["label"])
df["rec_color"]      = df["drift_score"].apply(lambda s: retrain_recommendation(s)["color"])
df["rec_emoji"]      = df["drift_score"].apply(lambda s: retrain_recommendation(s)["emoji"])

latest = df.sort_values("run_date").groupby("feature").last().reset_index()


# ─────────────────────────────────────────
# Cabeçalho
# ─────────────────────────────────────────

overall_status = latest["overall_status"].iloc[0]
status_color   = "#ef4444" if overall_status == "FAIL" else "#22c55e"
status_emoji   = "🔴" if overall_status == "FAIL" else "✅"

badge_class = "badge-github" if fonte == "github" else "badge-local"
badge_label = "🐙 GitHub API" if fonte == "github" else "📁 Local"

col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("🔍 Drift Lens")
    st.markdown(
        f"<span class='source-badge {badge_class}'>{badge_label}</span>"
        f"&nbsp; Última verificação: `{str(df['run_date'].max())[:19]}`",
        unsafe_allow_html=True
    )
with col_status:
    st.markdown(
        f"<div style='text-align:right;font-size:1.4rem;font-weight:700;color:{status_color}'>"
        f"{status_emoji} Pipeline: {overall_status}</div>",
        unsafe_allow_html=True
    )

st.divider()


# ─────────────────────────────────────────
# KPIs por feature
# ─────────────────────────────────────────

st.subheader("🎯 Drift Score por Feature")
st.caption("Score combinado de PSI, KS e KL-Divergence — indica probabilidade estimada de drift")

cols = st.columns(len(latest))
for i, (_, row) in enumerate(latest.iterrows()):
    rec  = retrain_recommendation(row["drift_score"])
    pct  = int(row["drift_score"] * 100)
    with cols[i]:
        st.metric(
            label=f"{rec['emoji']} {row['feature']}",
            value=f"{pct}%",
            delta=f"PSI={row['psi']:.3f}",
            delta_color="inverse"
        )
        st.caption(rec["label"])
        st.progress(row["drift_score"])


# ─────────────────────────────────────────
# Tabela de métricas
# ─────────────────────────────────────────

st.divider()
st.subheader("📋 Métricas Estatísticas por Feature")

def color_sev(val):
    c = {"CRÍTICO": "#ef444430", "MODERADO": "#f59e0b30", "ESTÁVEL": "#22c55e30"}
    return f"background-color: {c.get(val, 'transparent')}"

table = latest[["feature","psi","ks_p_value","kl_divergence","drift_score","severity","recommendation"]].copy()
table.columns = ["Feature","PSI","KS p-valor","KL Divergence","Drift Score","Severidade","Recomendação"]
table = table.round(4)

st.dataframe(
    table.style.map(color_sev, subset=["Severidade"]),
    use_container_width=True,
    hide_index=True
)

with st.expander("📖 Como interpretar"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**PSI**\n- < {psi_warn} → 🟢 Estável\n- {psi_warn}–{psi_crit} → 🟡 Moderado\n- > {psi_crit} → 🔴 Crítico")
    with c2:
        st.markdown(f"**KS Test**\n- p > {ks_alpha} → 🟢 Sem drift\n- p ≤ {ks_alpha} → 🔴 Drift detectado")
    with c3:
        st.markdown("**Drift Score**\n- 0–30% → 🟢 Baixo risco\n- 30–60% → 🟡 Moderado\n- 60–100% → 🔴 Retreinar")


# ─────────────────────────────────────────
# Gauge geral
# ─────────────────────────────────────────

st.divider()
st.subheader("🌡️ Drift Score Geral do Modelo")

overall_score = float(latest["drift_score"].mean())
rec_geral     = retrain_recommendation(overall_score)

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=overall_score * 100,
    number={"suffix": "%", "font": {"size": 40}},
    title={"text": f"{rec_geral['emoji']} {rec_geral['label']}", "font": {"size": 16}},
    gauge={
        "axis":  {"range": [0, 100], "ticksuffix": "%"},
        "bar":   {"color": rec_geral["color"]},
        "steps": [
            {"range": [0,  30], "color": "#dcfce7"},
            {"range": [30, 60], "color": "#fef9c3"},
            {"range": [60,100], "color": "#fee2e2"}
        ],
        "threshold": {"line": {"color": "#ef4444", "width": 3}, "thickness": 0.75, "value": 60}
    }
))
fig_gauge.update_layout(height=280, margin=dict(t=40, b=10))
st.plotly_chart(fig_gauge, use_container_width=True)


# ─────────────────────────────────────────
# Série temporal (só se tiver histórico da API)
# ─────────────────────────────────────────

if fonte == "github" and df["run_number"].nunique() > 1:
    st.divider()
    st.subheader("📈 Histórico de Drift Score — Runs do GitHub Actions")

    fig_hist = px.line(
        df, x="run_date", y="drift_score", color="feature", markers=True,
        labels={"run_date": "Data do Run", "drift_score": "Drift Score", "feature": "Feature"}
    )
    fig_hist.add_hline(y=0.6, line_dash="dash", line_color="#ef4444",
                       annotation_text="Limiar retreinamento (60%)")
    fig_hist.add_hline(y=0.3, line_dash="dot",  line_color="#f59e0b",
                       annotation_text="Limiar atenção (30%)")
    fig_hist.update_layout(height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("📊 PSI por Feature — Série Temporal")
    fig_psi = px.bar(
        df, x="run_date", y="psi", color="feature", barmode="group",
        labels={"run_date": "Data", "psi": "PSI", "feature": "Feature"}
    )
    fig_psi.add_hline(y=psi_crit, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"PSI crítico ({psi_crit})")
    fig_psi.add_hline(y=psi_warn, line_dash="dot",  line_color="#f59e0b",
                      annotation_text=f"PSI moderado ({psi_warn})")
    fig_psi.update_layout(height=320)
    st.plotly_chart(fig_psi, use_container_width=True)

else:
    st.info(
        "💡 Configure `GITHUB_TOKEN`, `REPO_OWNER` e `REPO_NAME` no arquivo `.env` "
        "para visualizar o histórico temporal de runs do CI/CD."
    )


# ─────────────────────────────────────────
# Recomendação de ação
# ─────────────────────────────────────────

st.divider()
st.subheader("🛠️ Recomendação de Ação")

critical_features = latest[latest["severity"] == "CRÍTICO"]["feature"].tolist()

if overall_score >= 0.6:
    st.error(f"""
**🔴 Retreinamento recomendado**

Features com drift crítico: **{', '.join(critical_features) if critical_features else 'múltiplas'}**

1. Verificar se a distribuição dos dados de produção mudou estruturalmente
2. Coletar novos dados rotulados das features afetadas
3. Retreinar o modelo com dados mais recentes
4. Atualizar `data/reference/train_data.csv` com os novos dados de referência
5. Fazer push → o CI/CD vai verificar se o drift foi resolvido
""")
elif overall_score >= 0.3:
    st.warning(f"""
**🟡 Monitoramento intensificado recomendado**

Features com drift moderado: **{', '.join(critical_features) if critical_features else 'algumas'}**

1. Aumentar frequência de coleta de dados de produção
2. Investigar causas do drift (sazonalidade? mudança de comportamento?)
3. Avaliar se as métricas de negócio já foram afetadas
""")
else:
    st.success("**🟢 Modelo estável — nenhuma ação necessária.**\n\nContinue monitorando conforme o agendamento do pipeline CI/CD.")


# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────

st.divider()
st.caption("Drift Lens · Monitor de Data Drift · PSI, KS-test, KL-Divergence · GitHub Actions CI/CD ·  UFPB ")