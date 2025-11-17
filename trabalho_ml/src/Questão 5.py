"""
Diagnóstico de Doenças Cardíacas - Análise Completa
Dataset: Heart Disease UCI
Modelos: Random Forest e SVM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_curve, auc, roc_auc_score, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
# ============================================================================

def load_heart_disease_data():
    """
    Carrega o dataset de doenças cardíacas
    Caso não tenha o arquivo local, usar dados simulados baseados no UCI dataset
    """
    # Colunas do dataset Heart Disease UCI
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    # URL do dataset (alternativa)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    try:
        df = pd.read_csv(url, names=columns, na_values='?')
        print("✓ Dataset carregado com sucesso!")
    except:
        print("⚠ Gerando dados simulados baseados no padrão UCI...")
        np.random.seed(42)
        n_samples = 303
        df = pd.DataFrame({
            'age': np.random.randint(29, 77, n_samples),
            'sex': np.random.choice([0, 1], n_samples),
            'cp': np.random.choice([0, 1, 2, 3], n_samples),
            'trestbps': np.random.randint(94, 200, n_samples),
            'chol': np.random.randint(126, 564, n_samples),
            'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'restecg': np.random.choice([0, 1, 2], n_samples),
            'thalach': np.random.randint(71, 202, n_samples),
            'exang': np.random.choice([0, 1], n_samples),
            'oldpeak': np.random.uniform(0, 6.2, n_samples),
            'slope': np.random.choice([0, 1, 2], n_samples),
            'ca': np.random.choice([0, 1, 2, 3], n_samples),
            'thal': np.random.choice([0, 1, 2, 3], n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.54, 0.46])
        })
    
    return df

def explore_data(df):
    """Análise exploratória dos dados"""
    print("\n" + "="*70)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("="*70)
    
    print(f"\n📊 Dimensões: {df.shape[0]} pacientes, {df.shape[1]} variáveis")
    print(f"\n🎯 Distribuição da variável alvo:")
    print(df['target'].value_counts())
    print(f"\n   Proporção: {df['target'].value_counts(normalize=True).round(3)}")
    
    print(f"\n❓ Valores ausentes:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   Nenhum valor ausente detectado!")
    
    print(f"\n📈 Estatísticas descritivas:")
    print(df.describe().round(2))
    
    return df

# ============================================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================================

def preprocess_data(df):
    """Tratamento de valores ausentes e preparação dos dados"""
    print("\n" + "="*70)
    print("PRÉ-PROCESSAMENTO DOS DADOS")
    print("="*70)
    
    df_clean = df.copy()
    
    # Tratamento de valores ausentes
    if df_clean.isnull().sum().sum() > 0:
        print("\n🔧 Tratando valores ausentes...")
        # Para variáveis numéricas: mediana
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                print(f"   - {col}: preenchido com mediana")
    
    # Convertendo target para binário (0 = sem doença, 1 = com doença)
    df_clean['target'] = (df_clean['target'] > 0).astype(int)
    
    print("\n✓ Pré-processamento concluído!")
    print(f"   Total de registros: {len(df_clean)}")
    print(f"   Total de features: {len(df_clean.columns) - 1}")
    
    return df_clean

def prepare_train_test(df):
    """Separação e normalização dos dados"""
    print("\n" + "="*70)
    print("PREPARAÇÃO DOS CONJUNTOS DE TREINO E TESTE")
    print("="*70)
    
    # Separar features e target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📦 Dados de treino: {X_train.shape[0]} amostras")
    print(f"📦 Dados de teste: {X_test.shape[0]} amostras")
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n✓ Dados normalizados (StandardScaler)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns, scaler

# ============================================================================
# 3. TREINAMENTO DOS MODELOS
# ============================================================================

def train_random_forest(X_train, y_train):
    """Treinamento do Random Forest com otimização"""
    print("\n" + "="*70)
    print("TREINAMENTO: RANDOM FOREST")
    print("="*70)
    
    # Grid Search para otimização
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    
    print("\n🔍 Executando Grid Search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Melhores parâmetros: {grid_search.best_params_}")
    print(f"✓ Melhor score (ROC-AUC): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_svm(X_train, y_train):
    """Treinamento do SVM com otimização"""
    print("\n" + "="*70)
    print("TREINAMENTO: SVM")
    print("="*70)
    
    # Grid Search para otimização
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    
    print("\n🔍 Executando Grid Search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Melhores parâmetros: {grid_search.best_params_}")
    print(f"✓ Melhor score (ROC-AUC): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# ============================================================================
# 4. AVALIAÇÃO DOS MODELOS
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name):
    """Avaliação completa do modelo"""
    print(f"\n{'='*70}")
    print(f"AVALIAÇÃO: {model_name.upper()}")
    print(f"{'='*70}")
    
    # Predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Sem Doença', 'Com Doença']))
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")
    
    return y_pred, y_pred_proba, roc_auc

def plot_comparison(rf_model, svm_model, X_test, y_test, feature_names):
    """Visualizações comparativas dos modelos"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Curvas ROC
    ax1 = plt.subplot(2, 3, 1)
    models = {'Random Forest': rf_model, 'SVM': svm_model}
    colors = {'Random Forest': 'blue', 'SVM': 'red'}
    
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[name], lw=2, 
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlabel('Taxa de Falsos Positivos')
    ax1.set_ylabel('Taxa de Verdadeiros Positivos')
    ax1.set_title('Curvas ROC - Comparação')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2 e 3. Matrizes de Confusão
    for idx, (name, model) in enumerate(models.items(), 2):
        ax = plt.subplot(2, 3, idx)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Matriz de Confusão - {name}')
        ax.set_ylabel('Valor Real')
        ax.set_xlabel('Valor Previsto')
    
    # 4. Feature Importance (Random Forest)
    ax4 = plt.subplot(2, 3, 4)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    ax4.barh(range(10), importances[indices], color='skyblue')
    ax4.set_yticks(range(10))
    ax4.set_yticklabels([feature_names[i] for i in indices])
    ax4.set_xlabel('Importância')
    ax4.set_title('Top 10 Features - Random Forest')
    ax4.invert_yaxis()
    
    # 5. Precision-Recall Curve
    ax5 = plt.subplot(2, 3, 5)
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ax5.plot(recall, precision, lw=2, label=name)
    
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title('Curvas Precision-Recall')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Comparação de Métricas
    ax6 = plt.subplot(2, 3, 6)
    metrics_comparison = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics_comparison.append({
            'Model': name,
            'Accuracy': report['accuracy'],
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score'],
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        })
    
    df_metrics = pd.DataFrame(metrics_comparison)
    x = np.arange(len(df_metrics.columns) - 1)
    width = 0.35
    
    for idx, row in df_metrics.iterrows():
        values = [row['Accuracy'], row['Precision'], row['Recall'], 
                 row['F1-Score'], row['ROC-AUC']]
        ax6.bar(x + idx*width, values, width, label=row['Model'])
    
    ax6.set_ylabel('Score')
    ax6.set_title('Comparação de Métricas')
    ax6.set_xticks(x + width / 2)
    ax6.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])
    ax6.legend()
    ax6.set_ylim([0, 1.1])
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('heart_disease_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráficos salvos em 'heart_disease_analysis.png'")
    plt.show()

# ============================================================================
# 5. ANÁLISE DE FEATURE IMPORTANCE
# ============================================================================

def analyze_feature_importance(rf_model, feature_names):
    """Análise detalhada da importância das features"""
    print("\n" + "="*70)
    print("ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS")
    print("="*70)
    
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n🔝 Top 10 Variáveis mais importantes:\n")
    for i, idx in enumerate(indices[:10], 1):
        print(f"   {i}. {feature_names[idx]:15s} : {importances[idx]:.4f}")
    
    return importances, indices

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ANÁLISE DE DIAGNÓSTICO DE DOENÇAS CARDÍACAS")
    print("="*70)
    
    # 1. Carregar e explorar dados
    df = load_heart_disease_data()
    df = explore_data(df)
    
    # 2. Pré-processamento
    df_clean = preprocess_data(df)
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_train_test(df_clean)
    
    # 3. Treinar modelos
    rf_model = train_random_forest(X_train, y_train)
    svm_model = train_svm(X_train, y_train)
    
    # 4. Avaliar modelos
    rf_pred, rf_proba, rf_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    svm_pred, svm_proba, svm_auc = evaluate_model(svm_model, X_test, y_test, "SVM")
    
    # 5. Análise de importância
    importances, indices = analyze_feature_importance(rf_model, feature_names)
    
    # 6. Visualizações
    plot_comparison(rf_model, svm_model, X_test, y_test, feature_names)
    
    # 7. Conclusões
    print("\n" + "="*70)
    print("CONCLUSÕES")
    print("="*70)
    
    if rf_auc > svm_auc:
        print(f"\n🏆 MELHOR MODELO: Random Forest")
        print(f"   ROC-AUC: {rf_auc:.4f} vs SVM: {svm_auc:.4f}")
        print(f"   Diferença: {(rf_auc - svm_auc):.4f}")
    else:
        print(f"\n🏆 MELHOR MODELO: SVM")
        print(f"   ROC-AUC: {svm_auc:.4f} vs Random Forest: {rf_auc:.4f}")
        print(f"   Diferença: {(svm_auc - rf_auc):.4f}")
    
    print("\n" + "="*70)
    print("✓ ANÁLISE CONCLUÍDA!")
    print("="*70)
