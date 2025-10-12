# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:19:11 2025

@author: Érica
"""

# ============================================================
#  RANDOM FOREST PARA PREVISÃO DE QUEIMADAS
# ============================================================

# 1. Importar bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------------------------------------
# 2. Carregar o dataset
# ------------------------------------------------------------
# Substitua o caminho abaixo pelo seu arquivo local
df = pd.read_csv("saida_merged_2019.csv")

# ------------------------------------------------------------
# 3. Criar variável-alvo binária
# ------------------------------------------------------------
# Vamos considerar risco >= 0.5 como “ocorrência de queimada”
df["queimada_ocorreu"] = (df["RiscoFogoMedia"] >= 0.5).astype(int)

# ------------------------------------------------------------
# 4. Selecionar variáveis explicativas
# ------------------------------------------------------------
features = [
    "temperatura_c",
    "umidade_relativa_percentual",
    "vento_direcao_grau",
    "vento_velocidade_ms",
    "precipitacao_mmdia",
    "latitude",
    "longitude"
]

X = df[features].fillna(df[features].median())   # Preencher valores ausentes
y = df["queimada_ocorreu"]

# ------------------------------------------------------------
# 5. Dividir dados em treino e teste
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 6. Criar e treinar o modelo Random Forest
# ------------------------------------------------------------
modelo_rf = RandomForestClassifier(
    n_estimators=200,       # número de árvores
    random_state=42,        # reprodutibilidade
    class_weight='balanced' # balanceia classes desiguais
)

modelo_rf.fit(X_train, y_train)

# ------------------------------------------------------------
# 7. Fazer previsões
# ------------------------------------------------------------
y_pred = modelo_rf.predict(X_test)

# ------------------------------------------------------------
# 8. Avaliar o modelo
# ------------------------------------------------------------
print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 9. Importância das variáveis
# ------------------------------------------------------------
importancias = pd.Series(modelo_rf.feature_importances_, index=features)
importancias = importancias.sort_values(ascending=False)

print("\nImportância das variáveis:\n")
print(importancias)

# ------------------------------------------------------------
# 10. Visualizar a importância das variáveis
# ------------------------------------------------------------
plt.figure(figsize=(8,4))
importancias.plot(kind='bar', color='forestgreen')
plt.title("Importância das Variáveis - Random Forest")
plt.ylabel("Importância Relativa")
plt.tight_layout()
plt.show()
