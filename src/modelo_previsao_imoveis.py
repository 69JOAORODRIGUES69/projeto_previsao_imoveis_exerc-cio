import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Carregar os dados
df = pd.read_csv('../data/house-prices.csv')

# 2. Preparar os dados
# Supondo que as colunas relevantes são 'LotArea' (área do terreno), 'OverallQual' (qualidade geral) e 'SalePrice' (preço de venda)
X = df[['LotArea', 'OverallQual']]  # Features (variáveis preditoras)
y = df['SalePrice']  # Target (variável alvo)

# 3. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 5. Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_test)

# 6. Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 7. Visualizar os resultados
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Regressão Linear: Preço de Imóveis')
plt.show()

