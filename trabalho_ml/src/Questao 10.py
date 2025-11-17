# =====================================================
# Previsão de Vendas Mensais em Rede de Varejo
# Modelos: Regressão Linear, Árvore de Decisão, XGBoost
# =====================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 1. Simulação do Dataset
# ======================
np.random.seed(42)
n_lojas = 50
meses = pd.date_range('2021-01-01', '2024-12-01', freq='MS')
dados = []

for loja in range(n_lojas):
    base_venda = np.random.randint(20000, 80000)
    for mes in meses:
        gasto_pub = np.random.randint(5000, 20000)
        funcionarios = np.random.randint(10, 50)
        promocao = np.random.choice([0, 1], p=[0.7, 0.3])
        mes_num = mes.month
        sazonal = np.sin(2 * np.pi * mes_num / 12)  # padrão sazonal
        ruido = np.random.normal(0, 3000)
        venda = base_venda + gasto_pub * 2.5 + funcionarios * 300 + promocao * 8000 + sazonal * 4000 + ruido
        dados.append([loja, mes, gasto_pub, funcionarios, promocao, mes_num, venda])

df = pd.DataFrame(dados, columns=["loja", "data", "gasto_pub", "funcionarios", "promocao", "mes", "vendas"])

# ======================
# 2. Feature Engineering
# ======================
df["trimestre"] = ((df["mes"] - 1) // 3) + 1

# ======================
# 3. Separação Treino/Teste
# ======================
X = df[["gasto_pub", "funcionarios", "promocao", "mes", "trimestre"]]
y = df["vendas"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================
# 4. Modelos
# ======================
modelos = {
    "Regressão Linear": LinearRegression(),
    "Árvore de Decisão": DecisionTreeRegressor(max_depth=6, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

resultados = {}

# ======================
# 5. Treinamento e Avaliação
# ======================
for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    preds = modelo.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    resultados[nome] = {"RMSE": rmse, "MAE": mae, "R²": r2}

# ======================
# 6. Resultados Comparativos
# ======================
resultados_df = pd.DataFrame(resultados).T.round(2)
print("=== MÉTRICAS DE DESEMPENHO ===")
print(resultados_df)

# ======================
# 7. Importância das Variáveis (XGBoost)
# ======================
xgb_model = modelos["XGBoost"]
importancias = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=importancias, y=importancias.index)
plt.title("Importância das Variáveis - XGBoost")
plt.show()

# ======================
# 8. Conclusão
# ======================
melhor_modelo = resultados_df["RMSE"].idxmin()
print(f"\n✅ O modelo com menor erro foi: {melhor_modelo}")
print("\n=== Principais variáveis preditoras (XGBoost) ===")
print(importancias)
