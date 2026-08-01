# 🔍 Drift Lens

Dashboard de monitoramento de **Data Drift** integrado a pipelines de CI/CD via GitHub Actions.

Desenvolvido como artefato do TCC — UFPB / Orbis Data.

---

## Estrutura do projeto

```
drift-lens/
├── .github/
│   └── workflows/
│       └── drift_detection.yml   # Pipeline CI/CD — roda a cada push
├── data/
│   ├── reference/
│   │   └── train_data.csv        # Dados de referência (treino) — fixo
│   └── production/
│       └── serving_data.csv      # Dados de produção — atualizado periodicamente
├── src/
│   ├── drift_metrics.py          # Calcula PSI, KS, KL e gera drift_report.json
│   ├── github_api.py             # Busca histórico de artefatos via GitHub API
│   └── generate_test_data.py     # Gera dados sintéticos com drift controlado
├── dashboard/
│   └── app.py                    # Dashboard Streamlit
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Ordem de execução local

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Gerar dados de teste com drift controlado
python src/generate_test_data.py

# 3. Calcular métricas de drift
python src/drift_metrics.py \
  --reference data/reference/train_data.csv \
  --production data/production/serving_data.csv \
  --output drift_report.json

# 4. Rodar o dashboard
streamlit run dashboard/app.py
```

---

## Variáveis de ambiente

Para ativar o histórico via GitHub API, configure:

```bash
GITHUB_TOKEN=ghp_...       # Personal Access Token com escopo repo
REPO_OWNER=mcarolina2      # Usuário do GitHub
REPO_NAME=drift-lens       # Nome do repositório
```

---

## Métricas implementadas

| Métrica | Tipo | Threshold crítico |
|---|---|---|
| PSI (Population Stability Index) | Estabilidade de distribuição | > 0.20 |
| KS-test (Kolmogorov-Smirnov) | Comparação de distribuições | p < 0.05 |
| KL-Divergence | Divergência de entropia | > 1.0 (normalizado) |

---

## Arquitetura do fluxo

```
dados de referência (treino)
        +
dados de produção (serving)
        ↓
drift_metrics.py → drift_report.json
        ↓
GitHub Actions → salva como artefato do run
        ↓
GitHub API → dashboard busca histórico de runs
        ↓
dashboard/app.py → exibe métricas + drift score + recomendação
```

