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
#%% Definición de funciones

def promedio(df, nombre_columna):
    avg = 0
    for i in range(df.shape[0]):
        avg += df.at[i, nombre_columna]
    avg /= df.shape[0]
    return avg

def desvio_estandar(df, nombre_columna):
    std = 0
    avg = promedio(df, nombre_columna)
    for i in range(df.shape[0]):
        std += (df.at[i, nombre_columna] - avg) ** 2
    std *= 1/(df.shape[0] - 1)
    std = np.sqrt(std)
    return std

def transformarAMatriz(avects):
    avects = np.array(avects)
    W = np.zeros((len(avects),len(avects[0])))
    for i in range(len(avects)):
        W[i] = avects[i].reshape(len(avects[0]))   
    return np.transpose(W)
#%% División de variables
# Ejercicio 1 b)

variable_dependiente = 'Customer_Segment'
Y = df_wine[[variable_dependiente]]
Xs = df_wine.drop(columns=variable_dependiente)

del variable_dependiente
#%% Estandarización de los datos
# Ejercicio 1 c)

x = df_wine['Customer_Segment']
# Dado un dataFrame y una columna, normaliza y centra la columna
def centrar_normalizar(df, nombre_columna):
    avg = promedio(df, nombre_columna)
    std = desvio_estandar(df, nombre_columna)

    for i in range(df.shape[0]):
        Xi = df.at[i, nombre_columna]
        df.at[i, nombre_columna] = (Xi-avg)/std

atributosXs = Xs.columns.values

for i in range(13):
    centrar_normalizar(Xs, atributosXs[i])

atributoY = Y.columns.values
centrar_normalizar(Y, atributoY[0])

del atributosXs, atributoY, i
#%% Matriz de covarianza
# Ejercicio 1 d)

def calculoCov(df):
    prom = np.mean(df, axis=0).values
    PROM = np.tile(prom.reshape((len(prom), 1)), df.shape[0])
    PROM = np.transpose(PROM)
    B = df.values
    B = B - PROM
    Mcov = np.dot(B.T,B)/ 178
    return Mcov

Mcov = calculoCov(Xs)
# armar la matriz de covarianza
matriz_de_cov = np.cov(np.transpose(Xs))
print(matriz_de_cov - Mcov)

del matriz_de_cov
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

A = Mcov
max_aval_cov = metodo_de_la_potencia(A)[0]
avect_asoc_cov = metodo_de_la_potencia(A)[1]
print('Aval máximo: ', max_aval_cov, 
      'Avect asociado : ', avect_asoc_cov)

del A
#%%
# Ejercicio 1 f)

# Dada una matriz simétrica y una cantidad n, devuelve los n autovalores de módulo máximo
# y sus correspondientes autovectores
def metodo_de_la_potencia_2(A,n):
    # Tomamos un vector cualquiera no nulo
    avect = np.random.rand(A.shape[0])
    avect = np.reshape(avect, (A.shape[0], 1))
    avects = []
    avals = []
    k = 9999
    for _ in range(n):
        for _ in range(k):
            avect = (A @ avect) / np.linalg.norm(A @ avect,2)
            aval = (np.transpose(avect) @ A @ avect) / (np.transpose(avect) @ avect)
        avects.append(avect)
        avals.append(aval)
        A = A - (aval * (avect @ np.transpose(avect)))
    avects = transformarAMatriz(avects)
    return avals, avects 

A = Mcov
n = 4
avals_cov = metodo_de_la_potencia_2(A,n)[0]
avects_asoc_cov = metodo_de_la_potencia_2(A,n)[1]
print('Avals máximos: ', avals_cov, 
      'Avects asociado : ', avects_asoc_cov)
print('Avals reales: ', np.linalg.eigh(A)[0],
      'Avects reales: ', np.linalg.eigh(A)[0])

del A, n
#%%
# Ejercicio 1 g)

def metodoDePCA(X, avects):
    X = X.values
    W = avects
    return X @ W

print(metodoDePCA(Xs, avects_asoc_cov))

def distDosVec(v,w):
    return np.linalg.norm(v-w,2)

def kNN(X_test,Y_train,X_train,k,n):
    avects_X_data = metodo_de_la_potencia_2(X_train.values,n)
    X_data = metodoDePCA(X_train, avects_X_data)
    Y_data = Y_train.values
    
    avects_X_pred = metodo_de_la_potencia_2(X_test.values,n)
    X_pred = metodoDePCA(X_test, avects_X_pred)
    Y_pred = np.zeros(X_pred.shape[0])
    
    for i in range(X_pred.shape[0]):
        k_vecinos = np.array(X_data.shape[0])
        for j in range(X_data.shape[0]):
            dist = distDosVec(X_pred[i], X_data[j])
            label = Y_data[j]
            k_vecinos[j] = (dist,label)
        # Ordenamos por el primer elemento de la tupla
        k_vecinos = k_vecinos[:k]
        # Contamos el label con más apariciones
        Y_pred[i] = #label con mas apariciones
        
    return Y_pred
    

lista = [(9,6),(1,2),(3,5),(6,7)]
lista = np.sort(lista)
#%%

# Centramos y estandarizamos df_wine y dividimos en datatest y data train
# usar stratify para tener una distribución?¿

# Hacer kNN de 1 a 4

# Llenar la tabla

# Graficar proyeccion T

#



