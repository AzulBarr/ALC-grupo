#%% Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
    for i in df[nombre_columna]:
       avg += i
    avg /= df.shape[0]
    return avg

def desvio_estandar(df, nombre_columna):
    std = 0
    avg = promedio(df, nombre_columna)
    for i in df[nombre_columna]:
        std += (i - avg) ** 2
    std *= 1/(df.shape[0] - 1)
    std = np.sqrt(std)
    return std

# Dado un dataFrame y una columna, normaliza y centra la columna
def centrar_normalizar(df, nombre_columna):
    avg = promedio(df, nombre_columna)
    std = desvio_estandar(df, nombre_columna)
    nuevosValores = []
    for i in df[nombre_columna]:
        Xi = i
        nuevosValores.append((Xi-avg)/std)
    df[nombre_columna] = nuevosValores

def transformarAMatriz(avects):
    avects = np.array(avects)
    W = np.zeros((len(avects),len(avects[0])))
    for i in range(len(avects)):
        W[i] = avects[i].reshape(len(avects[0]))   
    return np.transpose(W)

def distDosVec(v,w):
    return np.linalg.norm(v-w,2)

def claveParaOrdenar(tupla):
  return (tupla[0], -tupla[1])
#%% División de variables
# Ejercicio 1 b)

variable_dependiente = 'Customer_Segment'
Y = df_wine[[variable_dependiente]]
Xs = df_wine.drop(columns=variable_dependiente)

del variable_dependiente
#%% Estandarización de los datos
# Ejercicio 1 c)
atributosXs = Xs.columns.values

for i in range(13):
    centrar_normalizar(Xs, atributosXs[i])

del atributosXs, i
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

#%%
print(Mcov)

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
#%%
A = Mcov
max_aval_cov = metodo_de_la_potencia(A)[0]
avect_asoc_cov = metodo_de_la_potencia(A)[1]
#%%
print('Aval máximo: ', max_aval_cov, 
      'Avect asociado : ', avect_asoc_cov)
print('Avals reales: ', np.linalg.eigh(A)[0][12],
      'Avects reales: ', np.linalg.eigh(A)[1][12])

del A, max_aval_cov, avect_asoc_cov
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
        avals.append(aval[0][0])
        A = A - (aval * (avect @ np.transpose(avect)))
    avects = transformarAMatriz(avects)
    return avals, avects 
#%%
A = Mcov
n = 4
avals_cov = metodo_de_la_potencia_2(A,n)[0]
avects_asoc_cov = metodo_de_la_potencia_2(A,n)[1]
#%%
print('Avals máximos: ', avals_cov, 
      'Avects asociado : ', avects_asoc_cov)
print('Avals reales: ', np.linalg.eigh(A)[0],
      'Avects reales: ', np.linalg.eigh(A)[1])


del A, n, avals_cov, Mcov, avects_asoc_cov
#%%
# Ejercicio 1 g)

def metodoDePCA(X, n):
    cov = calculoCov(X)
    avects = metodo_de_la_potencia_2(cov,n)[1]
    X = X.values
    W = avects
    return X @ W
#%%
n = 4
print(metodoDePCA(Xs, n).shape)
#%%
def kNN(X_train,Y_train,X_test,k,n):
    # Calculamos la proyección de los datos de train con el PCA
    X_data = metodoDePCA(X_train, n)

    # Pasamos a matriz todo lo que sea df
    Y_data = Y_train.values[:,0]
    X_pred = X_test.values
    # Centramos y estandarizamos los datos de test
    for i in range(X_pred.shape[1]):
      X_pred[:, i] = (X_pred[:, i] - promedios[i])/desvios_estandares[i]

    # Proyectamos
    Mcov = calculoCov(X_train)
    avects_X_pred = metodo_de_la_potencia_2(Mcov,n)[1]
    X_pred = X_pred @ avects_X_pred
    Y_pred = np.zeros(X_pred.shape[0])

    labels = Y_train['Customer_Segment'].unique().tolist()

    for i in range(X_pred.shape[0]):
        k_vecinos = []
        for j in range(X_data.shape[0]):
            dist = distDosVec(X_pred[i], X_data[j])
            label = Y_data[j]
            k_vecinos.append((dist,label))
        # Ordenamos la lista de menor a mayor para obtener los que más cerca están
        k_vecinos = sorted(k_vecinos, key=claveParaOrdenar)
        k_vecinos = k_vecinos[:k]
        # Nos quedamos con los k vecinos
        segundos_elementos = [tupla[1] for tupla in k_vecinos]
        # Ahora armamos una nueva lista solo con los labels para saber cuál es el que más se repite
        repeticionesLabels = [segundos_elementos.count(labels[0]), segundos_elementos.count(labels[1]), segundos_elementos.count(labels[2])]
        Y_pred[i] = labels[repeticionesLabels.index(max(repeticionesLabels))]
        # Nos quedamos con el de mayor repeticiones

    return Y_pred
#%%
variable_dependiente = 'Customer_Segment'
Y = df_wine[[variable_dependiente]]
Xs = df_wine.drop(columns=variable_dependiente)

X_train, X_test, Y_train, Y_test = train_test_split(Xs, Y, test_size= 0.20, random_state= 5, stratify=Y)

atributosX_train = X_train.columns.values

promedios = []
desvios_estandares = []
for i in range(13):
    promedios.append(promedio(X_train, atributosX_train[i]))
    desvios_estandares.append(desvio_estandar(X_train, atributosX_train[i]))
    centrar_normalizar(X_train, atributosX_train[i])

#%%
k = 1
n = 4
Y_pred = kNN(X_train, Y_train, X_test, 1, 2)
print(Y_pred)
print(Y_pred - Y_test.values[:,0])


del k, n
#%%
# Armo tabla que nos piden
TABLA = pd.DataFrame(columns= ['Modelo PCA', 'Componente', 'Varianza explicada', 'Porcentaje %', 'Acumulado %'])

j = 0    
Mcov = calculoCov(X_train)
avals_cov = metodo_de_la_potencia_2(Mcov,13)[0]
suma = sum(avals_cov)
for i in range(1,5):

    if i == 1:
        TABLA.at[j,'Modelo PCA'] = '1 Componente Principal'
        TABLA.at[j, 'Componente'] = 1
        varianzaExplicada = avals_cov[0] / suma
        TABLA.at[j, 'Varianza explicada'] = varianzaExplicada
        TABLA.at[j, 'Porcentaje %'] = varianzaExplicada * 100
        TABLA.at[j, 'Acumulado %'] = varianzaExplicada * 100
        j += 1
        
    else:        
        varianzasExplicadas = []
        for k in range(1,i+1):
            varianzasExplicadas.append(avals_cov[k-1] / suma)
            TABLA.at[j,'Modelo PCA'] = f'{i} Componentes Principales'
            TABLA.at[j, 'Componente'] = k 
            TABLA.at[j, 'Varianza explicada'] = varianzasExplicadas[k-1]
            TABLA.at[j, 'Porcentaje %'] = varianzasExplicadas[k-1] * 100
            acumulado = sum(varianzasExplicadas)
            TABLA.at[j, 'Acumulado %'] = acumulado * 100
            j += 1

#%% 
TABLA.to_csv('Tabla.csv')
#%%
X1 = metodoDePCA(X_train, 1)
X1 = np.append(X1, Y_train.values, 1)
X2 = metodoDePCA(X_train, 2)
X2 = np.append(X2, Y_train.values, 1)
X3 = metodoDePCA(X_train, 3)
X3 = np.append(X3, Y_train.values, 1)
#%%
# Grafico con 1 componente principal
labels = Y_train['Customer_Segment'].unique().tolist()
for label in labels:
    X = X1[X1[:, 1] == label]
    plt.scatter(x=X[:, 0], y=np.zeros(X.shape[0]), label=f'Grupo {int(label)}')

#%%
# Grafico con 2 componentes principales

for label in labels:
    X = X2[X2[:, 2] == label]
    plt.scatter(x=X[:, 0], y=X[:, 1], label=f'Grupo {int(label)}')

#%%
# Grafico con 3 componentes principales

ax = plt.axes(projection='3d')
for label in labels:
    X = X3[X3[:, 3] == label]
    ax.scatter3D(xs=X[:,0], ys=X[:,1], zs=X[:,2], label=f'Grupo {int(label)}')



del j, i, TABLA, X1, X2, X3, Mcov, avals_cov, suma, k, varianzaExplicada, X
del varianzasExplicadas, acumulado, label, labels, ax
#%% 
del df_wine

