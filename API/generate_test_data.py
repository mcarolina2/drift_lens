"""
generate_test_data.py
Gera dados sintéticos: referência (treino) e produção com drift controlado.
Útil para validar o script de métricas e para o experimento do TCC.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000

os.makedirs("data/reference", exist_ok=True)
os.makedirs("data/production", exist_ok=True)

# ── Referência (distribuição original do treino) ──────────────────
reference = pd.DataFrame({
    "renda":       np.random.normal(5000, 1000, N),      # média 5000
    "idade":       np.random.normal(35, 8, N),           # média 35
    "score":       np.random.uniform(300, 850, N),       # uniforme
    "divida":      np.random.exponential(2000, N),       # exponencial
})

# ── Produção com drift controlado ────────────────────────────────
# renda: shift de média (+3000) → drift CRÍTICO (PSI > 0.2)
# idade: shift pequeno (+2)    → drift MODERADO
# score: sem mudança           → ESTÁVEL
# divida: mudança de escala    → drift CRÍTICO

production = pd.DataFrame({
    "renda":   np.random.normal(8000, 1000, N),          # drift severo
    "idade":   np.random.normal(37, 8, N),               # drift leve
    "score":   np.random.uniform(300, 850, N),           # sem drift
    "divida":  np.random.exponential(5000, N),           # drift severo
})


production.to_csv("../data/production/serving_data.csv", index=False)
reference.to_csv("../data/reference/train_data.csv", index=False)

print("✅ Dados gerados:")
print(f"   ../data/reference/train_data.csv     ({N} linhas)")
print(f"   ../data/production/serving_data.csv  ({N} linhas)")
print("\nDistribuição esperada dos resultados:")
print("  renda  → CRÍTICO   (shift de +3000 na média)")
print("  idade  → MODERADO  (shift de +2 na média)")
print("  score  → ESTÁVEL   (sem mudança)")
print("  divida → CRÍTICO   (escala exponencial dobrada)")