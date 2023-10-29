#%% Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%% Importación de datos
# Ejercicio 1 a)
# =============================================================================
# from google.colab import drive
# drive.mount('/content/drive') 
# df_wine = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/wine.csv')
# df_wine.head() 
# ============================================================================= 
df_wine = pd.read_csv('wine.csv')
df_wine.head()
#%% División de variables
# Ejercicio 1 b)
variable_dependiente = 'Customer_Segment'
Y = df_wine[[variable_dependiente]]
Xs = df_wine.drop(columns=variable_dependiente)

del variable_dependiente
#%% Normalización de los datos
# Ejercicio 1 c)
# Dado un dataFrame y una columna, normaliza la columna
def centrar_normalizar(df, nombre_columna):
    promedio = df[nombre_columna].mean()
    std = df[nombre_columna].std()

    for i in range(df.shape[0]):
        Xi = df.at[i, nombre_columna]
        df.at[i, nombre_columna] = (Xi-promedio)/std
atributosXs = Xs.columns.values

for i in range(13):
    centrar_normalizar(Xs, atributosXs[i])

atributoY = Y.columns.values
centrar_normalizar(Y, atributoY[0])

del atributosXs, atributoY, i
#%% Matriz de covarianza
# Ejercicio 1 d)
matriz_de_cov = np.cov(np.transpose(Xs))
print(matriz_de_cov)
#%%
# Ejercicio 1 e)
# Dada una matriz, devuelve el máximo autovalor y el correspondiente autovector
def metodo_de_la_potencia(A):
    # Tomamos un vector cualquiera no nulo
    avect = np.random.rand(A.shape[0])
    k = 9999
    for _ in range(k):
        avect = (A @ avect) / np.linalg.norm(A @ avect,2)
        aval = (np.transpose(avect) @ A @ avect) / (np.transpose(avect) @ avect)
    return aval, avect

A = matriz_de_cov
A = np.array([[-16, 32], [32, -64]])
print('Aval calculado: ', metodo_de_la_potencia(A)[0], 
      'Avect calculado : ', metodo_de_la_potencia(A)[1])

del A
#%%
# Ejercicio 1 f)
# Dada una matriz y una cantidad n, devuelve los n autovalores de módulo máximo
# y sus correspondientes autovectores
def metodo_de_la_potencia_2(A,n):
    # Tomamos un vector cualquiera no nulo
    avect = np.random.rand(A.shape[0])
    k = 9999
    for _ in range(n):
        for _ in range(k):
            avect = (A @ avect) / np.linalg.norm(A @ avect,2)
            aval = (np.transpose(avect) @ A @ avect) / (np.transpose(avect) @ avect)
        # completar
    return avect, aval
