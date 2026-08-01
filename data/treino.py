import os

os.makedirs("data/reference", exist_ok=True)
os.makedirs("data/production", exist_ok=True)

# Só salva se ainda não existir
if not os.path.exists("data/reference/train_data.csv"):
    reference.to_csv("data/reference/train_data.csv", index=False)
    print("✅ Referência criada")
else:
    print("⚠️  Referência já existe — pulando")

if not os.path.exists("data/production/serving_data.csv"):
    production.to_csv("data/production/serving_data.csv", index=False)
    print("✅ Produção criada")
else:
    print("⚠️  Produção já existe — pulando")