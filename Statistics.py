# Python program for calculate Bias and RMSE

# import
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

# open csv file
df = pd.read_csv('DadosTemperatura.csv')
df = df.round(2)

# dataframe for temperature data at each pressure level
# criando novo dataframe para armazenar dados por nivel de pressao
lst = list(range(60, 1050, 50))
dic = {chave: [[],[],[], []] for chave in lst} #cria um dicionario {P: [temperaturas]}

def nearests(valueP, valueN):
    return abs(valueN - valueP <= 10)


for pressure in df['P']:
    for key in dic.keys():
        if(nearests(pressure, key)):
            idx = df[df['P'] == pressure].index[0]
            tempP = df.iloc[idx]['T Sonda']
            tempI = df.iloc[idx]['T IASI']
            if tempI == '--':
                tempI = np.nan
            else:
                tempI = np.float64(tempI)
            dic[key][0].append(tempP)
            dic[key][1].append(tempI)


dfP = pd.DataFrame(data=dic)

# Statistics
def Bias(lstErrors):
    bias = np.mean(lstErrors)
    return bias

def RMSE(lstErrors):
    rmse = np.sqrt(np.mean([x**2 for x in lstErrors]))
    return rmse

for p in dfP.columns:
    lstErrors = []

    for i in range(len(dfP[p][0])):
        n = dfP[p][0][i] - dfP[p][1][i]  #observado(sonda) - estimado(iasi)
        lstErrors.append(n)
    cleanLstErrors = [x for x in lstErrors if not math.isnan(x)]

    x = Bias(cleanLstErrors)
    dfP[p][2].append(x)

    y = RMSE(cleanLstErrors)
    dfP[p][3].append(y)

#---- Bias
lstBias = []
for p in dfP.columns:
    n = dfP[p][2][0]
    lstBias.append(n)

#---- RMSE
lstRMSE = []
for p in dfP.columns:
    n = dfP[p][3][0]
    lstRMSE.append(n)

dfS = pd.DataFrame(list(zip(lst, lstBias, lstRMSE)), columns=['P', 'Bias', 'RMSE'])

# Plot
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(12,5))
plt.scatter(dfS['Bias'], dfS['P'], color='blue', label='Bias')
plt.plot(dfS['RMSE'], dfS['P'], color='green', label='RMSE')
plt.gca().invert_yaxis()
plt.xlim(np.nanmin(dfS['Bias'])-0.5, np.nanmax(dfS['RMSE'])+0.5)
plt.ylim(np.nanmin(dfS['P']-10), np.nanmax(dfS['P']+10))
plt.title('Temperature Nonlinear retrievals for IASI-MetOp')
plt.xlabel('BIAS/RMSE (K)')
plt.ylabel('P (hPa)')
plt.legend()
plt.savefig('BiasRMSE_20082011', dpi=400)
plt.show()

