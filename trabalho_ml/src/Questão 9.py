"""
Classificação de Imagens de Raio-X - Detecção de Pneumonia
Dataset: Chest X-ray (Pneumonia)
Arquitetura: Rede Neural Convolucional (CNN)
Técnicas: Data Augmentation, Transfer Learning, Early Stopping
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# TensorFlow e Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponível: {tf.config.list_physical_devices('GPU')}")

# Configurações
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

# ============================================================================
# 1. PREPARAÇÃO DOS DADOS
# ============================================================================

def create_data_generators():
    """
    Cria geradores de dados com data augmentation
    """
    print("\n" + "="*70)
    print("PREPARAÇÃO DOS DADOS")
    print("="*70)
    
    # Data Augmentation para treino
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% para validação
    )
    
    # Apenas rescaling para teste
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("\n✓ Data Augmentation configurado:")
    print("   - Rotação: ±20°")
    print("   - Deslocamento: ±20%")
    print("   - Zoom: ±20%")
    print("   - Flip horizontal")
    print("   - Shear: ±20%")
    
    return train_datagen, test_datagen

def simulate_data_structure():
    """
    Simula estrutura de dados para demonstração
    """
    print("\n📁 Estrutura esperada dos dados:")
    print("""
    chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
    """)
    
    print("⚠️  Para execução real, baixe o dataset do Kaggle:")
    print("    https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")

# ============================================================================
# 2. ARQUITETURA DO MODELO
# ============================================================================

def build_custom_cnn():
    """
    Constrói uma CNN customizada do zero
    """
    print("\n" + "="*70)
    print("MODELO 1: CNN CUSTOMIZADA")
    print("="*70)
    
    model = models.Sequential([
        # Bloco 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Camadas densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print("\n✓ Arquitetura da CNN:")
    model.summary()
    
    return model

def build_transfer_learning_model(base_model_name='VGG16'):
    """
    Constrói modelo com Transfer Learning
    """
    print(f"\n" + "="*70)
    print(f"MODELO 2: TRANSFER LEARNING - {base_model_name}")
    print("="*70)
    
    # Carregar modelo pré-treinado
    if base_model_name == 'VGG16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    
    # Congelar camadas base
    base_model.trainable = False
    
    # Adicionar camadas customizadas
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"\n✓ Transfer Learning com {base_model_name}")
    print(f"   Total de camadas: {len(model.layers)}")
    print(f"   Camadas treináveis: {sum([1 for layer in model.layers if layer.trainable])}")
    
    return model

# ============================================================================
# 3. CALLBACKS E TREINAMENTO
# ============================================================================

def get_callbacks(model_name):
    """
    Configura callbacks para treinamento
    """
    callbacks = [
        # Early Stopping
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Redução de learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Salvar melhor modelo
        ModelCheckpoint(
            f'best_model_{model_name}.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

def train_model_simulation():
    """
    Simulação de treinamento com dados fictícios
    """
    print("\n" + "="*70)
    print("SIMULAÇÃO DE TREINAMENTO")
    print("="*70)
    
    # Criar dados simulados
    np.random.seed(42)
    
    # Simular histórico de treinamento
    epochs = 25
    history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': [],
        'auc': [],
        'val_auc': []
    }
    
    # Simular evolução do treinamento
    for epoch in range(epochs):
        # Acurácia aumentando com ruído
        train_acc = 0.5 + (0.45 * epoch / epochs) + np.random.normal(0, 0.02)
        val_acc = 0.5 + (0.38 * epoch / epochs) + np.random.normal(0, 0.03)
        
        # Loss diminuindo
        train_loss = 0.7 * np.exp(-epoch/10) + np.random.normal(0, 0.05)
        val_loss = 0.75 * np.exp(-epoch/12) + np.random.normal(0, 0.06)
        
        # AUC aumentando
        train_auc = 0.5 + (0.48 * epoch / epochs) + np.random.normal(0, 0.01)
        val_auc = 0.5 + (0.42 * epoch / epochs) + np.random.normal(0, 0.02)
        
        history['accuracy'].append(min(train_acc, 0.98))
        history['val_accuracy'].append(min(val_acc, 0.92))
        history['loss'].append(max(train_loss, 0.05))
        history['val_loss'].append(max(val_loss, 0.08))
        history['auc'].append(min(train_auc, 0.99))
        history['val_auc'].append(min(val_auc, 0.94))
    
    print("\n✓ Treinamento simulado concluído!")
    print(f"   Épocas: {epochs}")
    print(f"   Acurácia final (treino): {history['accuracy'][-1]:.4f}")
    print(f"   Acurácia final (validação): {history['val_accuracy'][-1]:.4f}")
    print(f"   AUC final (validação): {history['val_auc'][-1]:.4f}")
    
    return history

# ============================================================================
# 4. VISUALIZAÇÃO E AVALIAÇÃO
# ============================================================================

def plot_training_history(history):
    """
    Visualiza o histórico de treinamento
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Acurácia
    axes[0, 0].plot(history['accuracy'], label='Treino', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Validação', linewidth=2)
    axes[0, 0].set_title('Acurácia ao Longo do Treinamento', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Acurácia')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Treino', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Validação', linewidth=2)
    axes[0, 1].set_title('Loss ao Longo do Treinamento', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(history['auc'], label='Treino', linewidth=2)
    axes[1, 0].plot(history['val_auc'], label='Validação', linewidth=2)
    axes[1, 0].set_title('AUC ao Longo do Treinamento', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting Analysis
    axes[1, 1].plot(
        np.array(history['accuracy']) - np.array(history['val_accuracy']),
        label='Gap Acurácia',
        linewidth=2
    )
    axes[1, 1].plot(
        np.array(history['val_loss']) - np.array(history['loss']),
        label='Gap Loss',
        linewidth=2
    )
    axes[1, 1].set_title('Análise de Overfitting', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Diferença (Val - Train)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráficos de treinamento salvos em 'training_history.png'")
    plt.show()

def evaluate_model_simulation():
    """
    Simula avaliação do modelo
    """
    print("\n" + "="*70)
    print("AVALIAÇÃO DO MODELO")
    print("="*70)
    
    # Simular predições
    np.random.seed(42)
    n_test = 624  # Tamanho típico do conjunto de teste
    
    # Gerar predições simuladas (modelo razoavelmente bom)
    y_true = np.random.choice([0, 1], n_test, p=[0.4, 0.6])
    y_pred_proba = []
    
    for true_label in y_true:
        if true_label == 1:
            # Pneumonia: maior probabilidade de predição correta
            prob = np.random.beta(8, 2)
        else:
            # Normal: maior probabilidade de predição correta
            prob = np.random.beta(2, 8)
        y_pred_proba.append(prob)
    
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Métricas
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['NORMAL', 'PNEUMONIA'],
                                digits=4))
    
    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Matriz de Confusão
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    axes[0].set_title('Matriz de Confusão', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Valor Real')
    axes[0].set_xlabel('Valor Previsto')
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Taxa de Falsos Positivos')
    axes[1].set_ylabel('Taxa de Verdadeiros Positivos')
    axes[1].set_title('Curva ROC', fontsize=12, fontweight='bold')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráficos de avaliação salvos em 'model_evaluation.png'")
    plt.show()
    
    return y_true, y_pred, y_pred_proba, roc_auc

# ============================================================================
# 5. DESAFIOS E MELHORIAS
# ============================================================================

def discuss_challenges():
    """
    Discute desafios e estratégias de melhoria
    """
    print("\n" + "="*70)
    print("DESAFIOS ENCONTRADOS NO TREINAMENTO")
    print("="*70)
    
    challenges = {
        "1. Desbalanceamento de Classes": [
            "- Dataset tipicamente tem mais casos de pneumonia que normais",
            "- Solução: Class weights, SMOTE, ou ajustar threshold"
        ],
        "2. Overfitting": [
            "- Modelo memoriza dados de treino",
            "- Solução: Dropout, Data Augmentation, Early Stopping"
        ],
        "3. Dados Limitados": [
            "- Conjuntos médicos são geralmente pequenos",
            "- Solução: Transfer Learning, Data Augmentation agressivo"
        ],
        "4. Qualidade das Imagens": [
            "- Variação em equipamentos e técnicas",
            "- Solução: Pré-processamento robusto, normalização"
        ],
        "5. Tempo de Treinamento": [
            "- CNNs profundas são computacionalmente custosas",
            "- Solução: GPU, modelos menores, mixed precision"
        ]
    }
    
    for challenge, solutions in challenges.items():
        print(f"\n{challenge}")
        for solution in solutions:
            print(f"  {solution}")

def suggest_improvements():
    """
    Sugere melhorias para o modelo
    """
    print("\n" + "="*70)
    print("ESTRATÉGIAS PARA MELHORAR O DESEMPENHO")
    print("="*70)
    
    improvements = {
        "🔧 Arquitetura": [
            "- Experimentar arquiteturas mais modernas (EfficientNet, DenseNet)",
            "- Ajustar número e tamanho das camadas",
            "- Usar blocos residuais (ResNet)"
        ],
        "📊 Dados": [
            "- Coletar mais dados de diferentes fontes",
            "- Usar técnicas de augmentation mais sofisticadas (Mixup, CutMix)",
            "- Balancear dataset com técnicas de sampling"
        ],
        "🎯 Treinamento": [
            "- Fine-tuning progressivo (unfreeze gradualmente)",
            "- Learning rate scheduling mais elaborado (Cosine Annealing)",
            "- Usar ensemble de múltiplos modelos"
        ],
        "🔍 Validação": [
            "- Cross-validation estratificada",
            "- Análise de casos mal classificados",
            "- Teste com dados externos (generalização)"
        ],
        "⚡ Performance": [
            "- Mixed precision training (FP16)",
            "- Otimização de hiperparâmetros (Optuna, Keras Tuner)",
            "- Gradient accumulation para batches maiores"
        ]
    }
    
    for category, suggestions in improvements.items():
        print(f"\n{category}")
        for suggestion in suggestions:
            print(f"  {suggestion}")

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================

def main():
    print("\n" + "="*70)
    print("CLASSIFICAÇÃO DE RAIO-X COM CNN - DETECÇÃO DE PNEUMONIA")
    print("="*70)
    
    # 1. Preparação de dados
    train_datagen, test_datagen = create_data_generators()
    simulate_data_structure()
    
    # 2. Construir modelos
    model_custom = build_custom_cnn()
    model_transfer = build_transfer_learning_model('VGG16')
    
    # 3. Callbacks
    callbacks = get_callbacks('cnn_custom')
    
    # 4. Treinamento (simulado)
    history = train_model_simulation()
    
    # 5. Visualizações
    plot_training_history(history)
    
    # 6. Avaliação
    y_true, y_pred, y_pred_proba, roc_auc = evaluate_model_simulation()
    
    # 7. Análise
    discuss_challenges()
    suggest_improvements()
    
    # 8. Resumo final
    print("\n" + "="*70)
    print("RESUMO DOS RESULTADOS")
    print("="*70)
    print(f"\n✓ Acurácia final: {history['val_accuracy'][-1]:.2%}")
    print(f"✓ AUC-ROC: {roc_auc:.4f}")
    print(f"✓ Loss final: {history['val_loss'][-1]:.4f}")
    
    print("\n📌 Recomendações:")
    print("   1. Usar Transfer Learning para melhor performance inicial")
    print("   2. Implementar data augmentation extensivo")
    print("   3. Monitorar overfitting com early stopping")
    print("   4. Considerar ensemble de múltiplos modelos")
    print("   5. Validar com dados de diferentes fontes")
    
    print("\n" + "="*70)
    print("✓ ANÁLISE CONCLUÍDA!")
    print("="*70)

if __name__ == "__main__":
    main()
