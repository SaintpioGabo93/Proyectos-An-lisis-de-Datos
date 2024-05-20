       # ------ Paso 1: Preprocesamiento de Datos ----------- #


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print(f'{X_train}/// {y_train} /// ---- ///{X_test}/// {y_test}')

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train) # Lo dejamos así porque todas las variables independientes necesitan de este método
X_test = sc.transform(X_test)

print(X_train)
print("")
print(X_test)


        # ------ Paso 2: Entrenamiento ----------- #


from sklearn.linear_model import LogisticRegression

lg_clasificador = LogisticRegression(random_state= 0)
lg_clasificador.fit(X_train, y_train)


        # ------ Paso 3: Predicción ------ #

decision_predicha = lg_clasificador.predict(sc.transform([[30,150000]]))
print(decision_predicha)

"""
vectores_concatenados = np.concatenate((vector_predicciones.reshape(len(vector_predicciones), 1),
                                         y_test.reshape(len(y_test), 1)),
                                         1) 
        
        """
y_pred = lg_clasificador.predict(X_test)
vector_concatenado = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

print(f'\n{vector_concatenado}\n')

        # ------ Paso 4: Matríz de Confusión --------- #


from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test,y_pred)
print(f'El segundo cuadrante de la matríz de confusión es para las predicciones correctas de la clase 0\n'
      f'El cuarto cuadrante es para las predicciones correctas de la clase 1\n'
      f'El primer cuadrante para las predicciones de la clase 1\n'
      f'El tercer cuadrante para las predicciones de la clase 0\n'
      f'{cm}')

print(f'La exactitud se obtiene del cociente entre las predicciones cociente entre las predicciones correctas sobre'
      f'el número total de muestras:\n'
      f'{acc}')

        # ------- Paso 5: Visulación los resultados del conjunto de entrenamiento -------- #


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, lg_clasificador.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regresión Logística (Training set)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, lg_clasificador.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regresión Logística (Test set)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()
