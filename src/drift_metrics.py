"""
drift_metrics.py
Calcula PSI, KS-test e KL-Divergence entre dados de referência e produção.
Salva resultado em drift_report.json para ser consumido pelo dashboard e CI/CD.

Uso:
    python src/drift_metrics.py \
        --reference data/reference/train_data.csv \
        --production data/production/serving_data.csv \
        --output drift_report.json
"""

import json
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from datetime import datetime, UTC


# ─────────────────────────────────────────
# 1. Métricas estatísticas
# ─────────────────────────────────────────

def calculate_psi(reference: np.ndarray, production: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.10 → estável
    PSI 0.10–0.20 → moderado
    PSI > 0.20 → crítico
    """
    breakpoints = np.linspace(
        min(reference.min(), production.min()),
        max(reference.max(), production.max()),
        buckets + 1
    )
    ref_counts, _  = np.histogram(reference,  bins=breakpoints)
    prod_counts, _ = np.histogram(production, bins=breakpoints)

    ref_pct  = np.where(ref_counts  == 0, 1e-6, ref_counts  / len(reference))
    prod_pct = np.where(prod_counts == 0, 1e-6, prod_counts / len(production))

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi), 6)


def calculate_ks(reference: np.ndarray, production: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov Test.
    p < 0.05 → distribuições significativamente diferentes (drift detectado)
    """
    statistic, p_value = ks_2samp(reference, production)
    return {
        "statistic":     round(float(statistic), 6),
        "p_value":       round(float(p_value), 6),
        "drift_detected": bool(p_value < 0.05)
    }


def calculate_kl_divergence(reference: np.ndarray, production: np.ndarray, buckets: int = 10) -> float:
    """
    KL-Divergence (entropia relativa).
    KL = 0 → distribuições idênticas. Quanto maior, maior o drift.
    """
    breakpoints = np.linspace(
        min(reference.min(), production.min()),
        max(reference.max(), production.max()),
        buckets + 1
    )
    ref_counts, _  = np.histogram(reference,  bins=breakpoints)
    prod_counts, _ = np.histogram(production, bins=breakpoints)

    ref_pct  = np.where(ref_counts  == 0, 1e-6, ref_counts  / len(reference))
    prod_pct = np.where(prod_counts == 0, 1e-6, prod_counts / len(production))

    kl = np.sum(ref_pct * np.log(ref_pct / prod_pct))
    return round(float(kl), 6)


def classify_severity(psi: float, ks_drift: bool) -> str:
    if psi > 0.2 or ks_drift:
        return "CRÍTICO"
    elif psi > 0.1:
        return "MODERADO"
    return "ESTÁVEL"


# ─────────────────────────────────────────
# 2. Pipeline principal
# ─────────────────────────────────────────

def run_drift_analysis(reference_path: str, production_path: str, output_path: str):
    print(f"[drift] Carregando referência : {reference_path}")
    reference_df  = pd.read_csv(reference_path)

    print(f"[drift] Carregando produção   : {production_path}")
    production_df = pd.read_csv(production_path)

    numeric_cols = reference_df.select_dtypes(include=np.number).columns
    common_cols  = [c for c in numeric_cols if c in production_df.columns]

    report = {
        "generated_at":    datetime.now(UTC).isoformat(),
        "reference_rows":  len(reference_df),
        "production_rows": len(production_df),
        "features":        {}
    }

    overall_critical = False

    for col in common_cols:
        ref_vals  = reference_df[col].dropna().values
        prod_vals = production_df[col].dropna().values

        psi = calculate_psi(ref_vals, prod_vals)
        ks  = calculate_ks(ref_vals, prod_vals)
        kl  = calculate_kl_divergence(ref_vals, prod_vals)
        sev = classify_severity(psi, ks["drift_detected"])

        if sev == "CRÍTICO":
            overall_critical = True

        report["features"][col] = {
            "psi":              psi,
            "ks_statistic":     ks["statistic"],
            "ks_p_value":       ks["p_value"],
            "ks_drift_detected":ks["drift_detected"],
            "kl_divergence":    kl,
            "severity":         sev
        }

        print(f"  [{sev:8}] {col:30} PSI={psi:.4f}  KS_p={ks['p_value']:.4f}  KL={kl:.4f}")

    report["overall_status"] = "FAIL" if overall_critical else "PASS"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[drift] Relatório salvo em : {output_path}")
    print(f"[drift] Status geral       : {report['overall_status']}")

    if overall_critical:
        sys.exit(1)


# ─────────────────────────────────────────
# 3. Entrypoint
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula métricas de data drift")
    parser.add_argument("--reference",  required=True, help="CSV de referência (treino)")
    parser.add_argument("--production", required=True, help="CSV de produção (serving recente)")
    parser.add_argument("--output",     default="drift_report.json", help="Caminho do relatório JSON")
    args = parser.parse_args()

    run_drift_analysis(args.reference, args.production, args.output)