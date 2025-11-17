"""
Questão 6 - Previsão do Valor de Imóveis
Dataset: California Housing (Scikit-Learn)
Modelos: Regressão Linear, XGBoost e Redes Neurais (ANNs)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("QUESTÃO 6 - PREVISÃO DO VALOR DE IMÓVEIS")
print("="*70)

# 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
print("\n1. CARREGANDO DATASET CALIFORNIA HOUSING...")
housingData = fetch_california_housing()
df = pd.DataFrame(housingData.data, columns=housingData.feature_names)
df['price'] = housingData.target

print(f"\nDimensões do dataset: {df.shape}")
print(f"\nPrimeiras linhas:\n{df.head()}")
print(f"\nEstatísticas descritivas:\n{df.describe()}")
print(f"\nValores nulos: {df.isnull().sum().sum()}")

# 2. FEATURE ENGINEERING
print("\n2. APLICANDO FEATURE ENGINEERING...")

# Criando novas features
df['roomsPerHouse'] = df['AveRooms'] * df['AveOccup']
df['bedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
df['populationPerHouse'] = df['Population'] / df['HouseAge']
df['medIncSquared'] = df['MedInc'] ** 2
df['houseLoc'] = df['Latitude'] * df['Longitude']

# Feature de proximidade do oceano (distância euclidiana aproximada)
df['distanceToCoast'] = np.sqrt((df['Latitude'] - 34)**2 + (df['Longitude'] + 118)**2)

print(f"Novas features criadas: {df.shape[1] - len(housingData.feature_names) - 1}")
print(f"Features totais: {df.shape[1] - 1}")

# 3. PREPARAÇÃO DOS DADOS
print("\n3. PREPARANDO DADOS PARA TREINAMENTO...")

X = df.drop('price', axis=1)
y = df['price']

# Divisão treino/teste
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização
scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(XTrain)
XTestScaled = scaler.transform(XTest)

print(f"Treino: {XTrain.shape[0]} amostras")
print(f"Teste: {XTest.shape[0]} amostras")

# 4. TREINAMENTO DOS MODELOS
print("\n4. TREINANDO MODELOS...")
print("-"*70)

results = {}

# 4.1 REGRESSÃO LINEAR
print("\n4.1 REGRESSÃO LINEAR")
lrModel = LinearRegression()
lrModel.fit(XTrainScaled, yTrain)
lrPred = lrModel.predict(XTestScaled)

lrRmse = np.sqrt(mean_squared_error(yTest, lrPred))
lrR2 = r2_score(yTest, lrPred)
lrMae = mean_absolute_error(yTest, lrPred)

results['Linear Regression'] = {
    'RMSE': lrRmse,
    'R²': lrR2,
    'MAE': lrMae,
    'predictions': lrPred
}

print(f"RMSE: {lrRmse:.4f}")
print(f"R²: {lrR2:.4f}")
print(f"MAE: {lrMae:.4f}")

# 4.2 XGBOOST
print("\n4.2 XGBOOST")
xgbModel = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbosity=0
)
xgbModel.fit(XTrain, yTrain)
xgbPred = xgbModel.predict(XTest)

xgbRmse = np.sqrt(mean_squared_error(yTest, xgbPred))
xgbR2 = r2_score(yTest, xgbPred)
xgbMae = mean_absolute_error(yTest, xgbPred)

results['XGBoost'] = {
    'RMSE': xgbRmse,
    'R²': xgbR2,
    'MAE': xgbMae,
    'predictions': xgbPred
}

print(f"RMSE: {xgbRmse:.4f}")
print(f"R²: {xgbR2:.4f}")
print(f"MAE: {xgbMae:.4f}")

# 4.3 REDES NEURAIS ARTIFICIAIS (ANNs)
print("\n4.3 REDES NEURAIS ARTIFICIAIS")
annModel = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
annModel.fit(XTrainScaled, yTrain)
annPred = annModel.predict(XTestScaled)

annRmse = np.sqrt(mean_squared_error(yTest, annPred))
annR2 = r2_score(yTest, annPred)
annMae = mean_absolute_error(yTest, annPred)

results['Neural Network'] = {
    'RMSE': annRmse,
    'R²': annR2,
    'MAE': annMae,
    'predictions': annPred
}

print(f"RMSE: {annRmse:.4f}")
print(f"R²: {annR2:.4f}")
print(f"MAE: {annMae:.4f}")

# 5. COMPARAÇÃO DE RESULTADOS
print("\n" + "="*70)
print("5. COMPARAÇÃO DE RESULTADOS")
print("="*70)

resultsDf = pd.DataFrame(results).T
print(f"\n{resultsDf}")

# Identificar melhor modelo
bestModel = resultsDf['RMSE'].idxmin()
print(f"\n🏆 MELHOR MODELO: {bestModel}")
print(f"   RMSE: {resultsDf.loc[bestModel, 'RMSE']:.4f}")
print(f"   R²: {resultsDf.loc[bestModel, 'R²']:.4f}")
print(f"   MAE: {resultsDf.loc[bestModel, 'MAE']:.4f}")

# 6. OTIMIZAÇÃO DO MELHOR MODELO (XGBoost)
print("\n" + "="*70)
print("6. OTIMIZAÇÃO DO MODELO XGBOOST")
print("="*70)

print("\nRealizando Grid Search para otimização de hiperparâmetros...")

paramGrid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

gridSearch = GridSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0),
    paramGrid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0
)

gridSearch.fit(XTrain, yTrain)

# Melhor modelo otimizado
bestXgb = gridSearch.best_estimator_
bestXgbPred = bestXgb.predict(XTest)

bestXgbRmse = np.sqrt(mean_squared_error(yTest, bestXgbPred))
bestXgbR2 = r2_score(yTest, bestXgbPred)
bestXgbMae = mean_absolute_error(yTest, bestXgbPred)

print(f"\nMelhores hiperparâmetros encontrados:")
for param, value in gridSearch.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nMétricas do modelo otimizado:")
print(f"  RMSE: {bestXgbRmse:.4f} (melhoria: {((xgbRmse - bestXgbRmse)/xgbRmse)*100:.2f}%)")
print(f"  R²: {bestXgbR2:.4f}")
print(f"  MAE: {bestXgbMae:.4f}")

# 7. ANÁLISE DE IMPORTÂNCIA DAS FEATURES
print("\n" + "="*70)
print("7. IMPORTÂNCIA DAS FEATURES (Top 10)")
print("="*70)

featureImportance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': bestXgb.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

print(f"\n{featureImportance.to_string(index=False)}")

# 8. VISUALIZAÇÕES
print("\n8. GERANDO VISUALIZAÇÕES...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: Comparação de modelos
ax1 = axes[0, 0]
models = list(results.keys())
rmseValues = [results[m]['RMSE'] for m in models]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax1.bar(models, rmseValues, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax1.set_title('Comparação de RMSE entre Modelos', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

# Gráfico 2: R² dos modelos
ax2 = axes[0, 1]
r2Values = [results[m]['R²'] for m in models]
bars = ax2.bar(models, r2Values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('R²', fontsize=12, fontweight='bold')
ax2.set_title('Comparação de R² entre Modelos', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

# Gráfico 3: Valores Reais vs Preditos (Melhor Modelo)
ax3 = axes[1, 0]
ax3.scatter(yTest, bestXgbPred, alpha=0.5, s=10)
ax3.plot([yTest.min(), yTest.max()], [yTest.min(), yTest.max()], 
         'r--', lw=2, label='Predição Perfeita')
ax3.set_xlabel('Valores Reais', fontsize=12, fontweight='bold')
ax3.set_ylabel('Valores Preditos', fontsize=12, fontweight='bold')
ax3.set_title('XGBoost Otimizado: Real vs Predito', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Gráfico 4: Importância das Features
ax4 = axes[1, 1]
topFeatures = featureImportance.head(8)
ax4.barh(topFeatures['Feature'], topFeatures['Importance'], color='#9b59b6', alpha=0.7)
ax4.set_xlabel('Importância', fontsize=12, fontweight='bold')
ax4.set_title('Top 8 Features Mais Importantes', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('questao6_resultados.png', dpi=300, bbox_inches='tight')
print("Gráficos salvos em 'questao6_resultados.png'")

# 9. CONCLUSÕES E RECOMENDAÇÕES
print("\n" + "="*70)
print("9. CONCLUSÕES E RECOMENDAÇÕES")
print("="*70)

print(f"""
RESPOSTAS ÀS PERGUNTAS:

1. Qual modelo teve menor erro de previsão?
   → {bestModel} obteve o melhor desempenho com RMSE de {resultsDf.loc[bestModel, 'RMSE']:.4f}

2. Como otimizar ainda mais o desempenho?

   ESTRATÉGIAS IMPLEMENTADAS:
   ✓ Feature Engineering: Criação de 7 novas features relevantes
   ✓ Grid Search: Otimização de hiperparâmetros do XGBoost
   ✓ Normalização: Padronização dos dados para ANNs
   
   PRÓXIMAS ESTRATÉGIAS RECOMENDADAS:
   • Ensemble Methods: Combinar predições de múltiplos modelos
   • Feature Selection: Remover features pouco importantes
   • Engenharia de Features Avançada: Interações e transformações não-lineares
   • Cross-Validation: Validação cruzada mais robusta (k-fold)
   • Regularização: Testar diferentes valores de alpha para ANNs
   • Hiperparâmetros ANNs: Testar diferentes arquiteturas de rede
   • Tratamento de Outliers: Análise e tratamento de valores extremos
   • Dados Geográficos: Incorporar informações mais detalhadas de localização

   GANHO COM OTIMIZAÇÃO:
   • Redução de {((xgbRmse - bestXgbRmse)/xgbRmse)*100:.2f}% no RMSE do XGBoost
   • R² aumentou de {xgbR2:.4f} para {bestXgbR2:.4f}
""")

print("\n" + "="*70)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("="*70)
