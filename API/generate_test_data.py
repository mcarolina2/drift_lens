import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # pasta API/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # sobe para drift_lens/

# ── Referência (distribuição original do treino) ──────────────────
reference = pd.DataFrame({
    "renda":       np.random.normal(5000, 1000, N),
    "idade":       np.random.normal(35, 8, N),
    "score":       np.random.uniform(300, 850, N),
    "divida":      np.random.exponential(2000, N),
})

# ── Produção com drift controlado ────────────────────────────────
production = pd.DataFrame({
    "renda":   np.random.normal(8000, 1000, N),
    "idade":   np.random.normal(37, 8, N),
    "score":   np.random.uniform(300, 850, N),
    "divida":  np.random.exponential(5000, N),
})

# ── Salvar (caminhos absolutos, independentes de onde o script roda) ──
production.to_csv(
    os.path.join(PROJECT_ROOT, "data", "production", "serving_data.csv"),
    index=False
)
reference.to_csv(
    os.path.join(PROJECT_ROOT, "data", "reference", "train_data.csv"),
    index=False
)

print("✅ Dados gerados:")
print(f"   data/reference/train_data.csv     ({N} linhas)")
print(f"   data/production/serving_data.csv  ({N} linhas)")
print("\nDistribuição esperada dos resultados:")
print("  renda  → CRÍTICO   (shift de +3000 na média)")
print("  idade  → MODERADO  (shift de +2 na média)")
print("  score  → ESTÁVEL   (sem mudança)")
print("  divida → CRÍTICO   (escala exponencial dobrada)")