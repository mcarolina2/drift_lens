"""
github_api.py
Busca histórico de artefatos de drift via GitHub REST API.
Calcula drift probability score combinando PSI, KS e KL.

Variáveis de ambiente necessárias:
    GITHUB_TOKEN  — Personal Access Token com escopo repo
    REPO_OWNER    — usuário do GitHub (ex: mcarolina2)
    REPO_NAME     — nome do repositório (ex: drift-lens)

Fluxo:
    1. list_drift_artifacts()       → lista artefatos do workflow
    2. download_artifact_json()     → baixa e descompacta cada artefato
    3. build_drift_history()        → monta DataFrame com série temporal
    4. load_local_report()          → fallback offline (drift_report.json local)
"""

import os
import io
import json
import zipfile
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
REPO_OWNER   = os.getenv("REPO_OWNER", "")
REPO_NAME    = os.getenv("REPO_NAME", "")

HEADERS = {
    "Authorization":        f"Bearer {GITHUB_TOKEN}",
    "Accept":               "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

BASE_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"


# ─────────────────────────────────────────
# 1. Listar artefatos
# ─────────────────────────────────────────

def list_drift_artifacts(max_pages: int = 5) -> list[dict]:
    """
    Retorna lista de artefatos cujo nome começa com 'drift-report',
    ordenados do mais antigo para o mais recente (série temporal).
    """
    artifacts = []
    page = 1

    while page <= max_pages:
        resp = requests.get(
            f"{BASE_URL}/actions/artifacts",
            headers=HEADERS,
            params={"per_page": 30, "page": page}
        )

        if resp.status_code == 401:
            raise PermissionError("Token inválido ou expirado. Verifique GITHUB_TOKEN.")
        if resp.status_code == 404:
            raise FileNotFoundError(f"Repositório {REPO_OWNER}/{REPO_NAME} não encontrado.")

        resp.raise_for_status()
        data  = resp.json()
        batch = data.get("artifacts", [])

        if not batch:
            break

        artifacts.extend([
            a for a in batch
            if a["name"].startswith("drift-report") and not a["expired"]
        ])
        page += 1

    artifacts.sort(key=lambda a: a["created_at"])
    return artifacts


# ─────────────────────────────────────────
# 2. Baixar artefato
# ─────────────────────────────────────────

def download_artifact_json(artifact: dict) -> dict | None:
    """
    Baixa o ZIP do artefato e extrai o drift_report.json em memória.
    Retorna o dicionário parseado ou None se falhar.
    """
    resp = requests.get(
        artifact["archive_download_url"],
        headers=HEADERS,
        stream=True
    )

    if resp.status_code != 200:
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for name in z.namelist():
            if name.endswith(".json"):
                with z.open(name) as f:
                    return json.load(f)

    return None


# ─────────────────────────────────────────
# 3. Drift Probability Score
# ─────────────────────────────────────────

def _normalize_psi(psi: float) -> float:
    """PSI → [0,1]. Satura em PSI=0.3 (drift severo)."""
    return min(psi / 0.3, 1.0)


def _normalize_ks(p_value: float) -> float:
    """KS p-valor → [0,1] invertido. p pequeno = drift alto."""
    return max(0.0, 1.0 - p_value)


def _normalize_kl(kl: float) -> float:
    """KL-Divergence → [0,1]. Satura em KL=1.0."""
    return min(kl / 1.0, 1.0)


def drift_probability_score(
    psi: float,
    ks_p_value: float,
    kl: float,
    w_psi: float = 0.5,
    w_ks:  float = 0.3,
    w_kl:  float = 0.2
) -> float:
    """
    Score combinado em [0,1] representando a probabilidade estimada de drift.

    Interpretação:
        0.0 – 0.30 → baixo risco    (modelo estável)
        0.30 – 0.60 → risco moderado (monitorar)
        0.60 – 1.00 → alto risco    (retreinamento recomendado)

    Pesos padrão:
        PSI = 0.5 (métrica mais usada em produção para data drift)
        KS  = 0.3 (teste estatístico formal)
        KL  = 0.2 (complementar ao PSI)
    """
    score = (
        w_psi * _normalize_psi(psi) +
        w_ks  * _normalize_ks(ks_p_value) +
        w_kl  * _normalize_kl(kl)
    )
    return round(score, 4)


def retrain_recommendation(score: float) -> dict:
    """Traduz o score em recomendação acionável para o dashboard."""
    if score >= 0.6:
        return {"label": "Retreinamento recomendado", "color": "#ef4444", "emoji": "🔴", "urgency": "alta"}
    elif score >= 0.3:
        return {"label": "Monitorar com atenção",     "color": "#f59e0b", "emoji": "🟡", "urgency": "média"}
    return     {"label": "Modelo estável",            "color": "#22c55e", "emoji": "🟢", "urgency": "baixa"}


# ─────────────────────────────────────────
# 4. Série temporal completa (GitHub API)
# ─────────────────────────────────────────

def build_drift_history(
    w_psi: float = 0.5,
    w_ks:  float = 0.3,
    w_kl:  float = 0.2
) -> pd.DataFrame:
    """
    Busca todos os artefatos de drift do repositório,
    calcula o drift probability score por feature e por run,
    e retorna um DataFrame com série temporal pronta para o dashboard.

    Colunas:
        run_date, run_number, feature, psi, ks_p_value, ks_drift,
        kl_divergence, severity, drift_score, recommendation,
        rec_color, rec_emoji, overall_status
    """
    if not GITHUB_TOKEN:
        raise EnvironmentError(
            "GITHUB_TOKEN não configurado. "
            "Crie um arquivo .env com GITHUB_TOKEN, REPO_OWNER e REPO_NAME."
        )

    artifacts = list_drift_artifacts()

    if not artifacts:
        return pd.DataFrame()

    rows = []
    for artifact in artifacts:
        report = download_artifact_json(artifact)
        if not report:
            continue

        run_date   = datetime.fromisoformat(artifact["created_at"].replace("Z", "+00:00"))
        run_number = artifact["name"].split("-")[-1]

        for feature, metrics in report["features"].items():
            score = drift_probability_score(
                metrics["psi"],
                metrics["ks_p_value"],
                metrics["kl_divergence"],
                w_psi, w_ks, w_kl
            )
            rec = retrain_recommendation(score)

            rows.append({
                "run_date":       run_date,
                "run_number":     int(run_number) if run_number.isdigit() else 0,
                "feature":        feature,
                "psi":            metrics["psi"],
                "ks_p_value":     metrics["ks_p_value"],
                "ks_drift":       metrics["ks_drift_detected"],
                "kl_divergence":  metrics["kl_divergence"],
                "severity":       metrics["severity"],
                "drift_score":    score,
                "recommendation": rec["label"],
                "rec_color":      rec["color"],
                "rec_emoji":      rec["emoji"],
                "overall_status": report["overall_status"]
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["run_date", "feature"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────
# 5. Fallback local
# ─────────────────────────────────────────

def load_local_report(
    path: str = "drift_report.json",
    w_psi: float = 0.5,
    w_ks:  float = 0.3,
    w_kl:  float = 0.2
) -> pd.DataFrame:
    """
    Carrega um único drift_report.json local.
    Usado quando GITHUB_TOKEN não está configurado (modo offline / desenvolvimento).
    """
    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    rows = []
    for feature, metrics in report["features"].items():
        score = drift_probability_score(
            metrics["psi"],
            metrics["ks_p_value"],
            metrics["kl_divergence"],
            w_psi, w_ks, w_kl
        )
        rec = retrain_recommendation(score)
        rows.append({
            "run_date":       report["generated_at"],
            "run_number":     0,
            "feature":        feature,
            "psi":            metrics["psi"],
            "ks_p_value":     metrics["ks_p_value"],
            "ks_drift":       metrics["ks_drift_detected"],
            "kl_divergence":  metrics["kl_divergence"],
            "severity":       metrics["severity"],
            "drift_score":    score,
            "recommendation": rec["label"],
            "rec_color":      rec["color"],
            "rec_emoji":      rec["emoji"],
            "overall_status": report["overall_status"]
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 6. Loader unificado (usado pelo dashboard)
# ─────────────────────────────────────────

def load_drift_data(
    w_psi: float = 0.5,
    w_ks:  float = 0.3,
    w_kl:  float = 0.2
) -> tuple[pd.DataFrame, str]:
    """
    Tenta carregar dados da GitHub API.
    Se não estiver configurado, cai para o arquivo local.

    Retorna:
        (DataFrame, fonte)  onde fonte é "github" ou "local"
    """
    if GITHUB_TOKEN and REPO_OWNER and REPO_NAME:
        try:
            df = build_drift_history(w_psi, w_ks, w_kl)
            if not df.empty:
                return df, "github"
        except Exception as e:
            print(f"[github_api] Aviso: falha na API ({e}). Usando arquivo local.")

    df = load_local_report(w_psi=w_psi, w_ks=w_ks, w_kl=w_kl)
    return df, "local"
