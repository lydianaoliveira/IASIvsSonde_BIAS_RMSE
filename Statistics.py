#

# import
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.evaluate import bias_variance_decomp

# abrindo o arquivo
df = pd.read_csv('DadosTemperatura.csv')
df = df.round(2)
print(df)

'''
iasiT = df['T IASI']
sondaT = df['T Sonda']

# Valores MAX/MIN e média
#--- IASI
iasiTMax = np.max(iasiT)
iasiTMin = np.min(iasiT)
iasiMedia = round(np.mean(iasiT), 2)
print(f'IASI valor máximo {iasiTMax}, valor mínimo {iasiTMin} e média {iasiMedia}')

#--- Sonda
sondaTMax = np.max(sondaT)
sondaTMin = np.min(sondaT)
sondaMedia = round(np.mean(sondaT), 2)
print(f'Sonda valor máximo {sondaTMax}, valor mínimo {sondaTMin} e média {sondaMedia}')

# Erro médio absoluto
MAE = mae(df['T Sonda'], df['T IASI'])
print(f'Erro Médio Absoluto {MAE}')

# Erro quadrático médio
MSE = mse(df['T Sonda'], df['T IASI'])
print(f'Erro Quadrático Médio {MSE}')

# Root Mean Square Error (RMSE)
RMSE = math.sqrt(MSE)
print(f'RMSE {RMSE}')

# BIAS
x = df['T IASI'].to_numpy()
y = df['T Sonda'].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = LinearRegression()
mse, bias, var = bias_variance_decomp(model, x_train, y_train, x_test, y_test, loss='mse', num_rounds=98, random_seed=1)

print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)

print(40*'-')
print('')

# Desvio padrão
df['Desvio'] = df.std(axis=1)

# Diferença
df['Diferença'] = df['T Sonda'] - df['T IASI']

print(df)
'''

# Salvando DataFrame em arquivo csv
#df.to_csv(r'DataTemperaturaIASISonda.csv', index=False)


