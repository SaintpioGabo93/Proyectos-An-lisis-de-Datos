        # ----- Paso 1: Importación de librerias y Creación del Conjunto de Datos ------------ #

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
''' 
Para procesar este texto, debemos poner nuevos parámetros.

Delimiter = este nos va a decir cuál es el caracter que se utiliza como separador de cada uno de nuestras palabras
quoting = este parámetro nos dice qué caracteres debe ignorar
'''
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting= 3)

        # ------ Paso 2: Limpieza del texto ---------------- #
"""
Este es un paso escencial para el procesamiento de lenguaje natural
"""
# Herramientas para la limpieza de texo
import re
import nltk # Esta librería nos va a remover todos los articulos en inglés como the, er, etc.

nltk.download('stopwords') # Utilizamos este método para remover todas las palabras que detienen

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # Este método funciona para remover las conjugaciones y mantener las raices de los verbos

# Comenzamos con la limpieza de texto

corpus = [] # En esta lista vamos a guardar las reseñas pero ya limpias.

'''
Vamos a crear un ciclo for para limpiar cada una de las reseñas y estas se van a guardar en la lista que creamos 
llamada corpus
'''

for i in range(0, 1000):# En el rango ponemos el total de reseñas que tenemos, y como tenemos 1000, ponemos ese número
    '''
    Con este método quitamos tod○ lo que no sean letras del alfabeto y se reemplazan por espacios en el conjunto de datos debajo de la columna Review
    '''
    resenia = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    resenia = resenia.lower()
    resenia = resenia.split() # Separamos en cada una de las palabras de la reseña

    ps = PorterStemmer()
    """
    A continuación se crea esa variable para que no quite la palabra not de las reseñas
    """
    todas_stopwords = stopwords.words('english') # Creamos la variable del idioma que vamos a utilizar para quitarle las stopwords
    todas_stopwords.remove('not') # Con esto evitamos que remueva el not
    """
    A continuación la variable resenia se va a actualizar con el ciclo for para que remueva las stopwords 
    """
    resenia = [ps.stem(palabra) for palabra in resenia if not palabra in set(todas_stopwords)]
    resenia = ' '.join(resenia)
    corpus.append(resenia)

print(corpus)

        # ----- Paso 3: Crear el modelo Bolsa de Palabras ------ #
"""
Para la creación del modelo, con los pasos anteriores, ya podemos crear nuestra matríz de características X además de
el vector de variables dependientes
"""
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features= 1500) # Sólo necesita un parámetro y es el número de palabras que se repitan mas, en este caso sólo utilizaremos las 1500 palabras más comúnes
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


# Separamos en un conjunto de entrenamiento y de prueba

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


        # ------ Paso 4: Entrenamiento --------- #

from sklearn.naive_bayes import GaussianNB

NB_clasificador = GaussianNB()
NB_clasificador.fit(X_train, y_train)


        # ----- Paso 5: Predicción de resultados -------- #


y_pred = NB_clasificador.predict(X_test)

prediccion = np.concatenate((y_pred.reshape(len(y_pred),1),
                             y_test.reshape(len(y_test),1)),
                            1) # 1 para que sea tensor columna

print(prediccion)


        # ----- Paso 6: Matríz de Confusión -------- #

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f'Matríz de Confusión:\n{cm}')
print(f'Precisión:\n{acc}')


'''
Bonus para hacer una sola predicción ya sea buena o mala, tenemos que realizar el mismo procedimiento de limpieza de 
datos en el nuevo texto. 

A continuación el código para hacer una sola predicción
'''

# Reseña nueva y proceso de limpieza de texto
nueva_resenia = 'I love this restaurant so much'
nueva_resenia = re.sub('[^a-zA-Z]', ' ', nueva_resenia)
nueva_resenia = nueva_resenia.lower()
nueva_resenia = nueva_resenia.split()
# Remoción de artículos y palabras innecesarias
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
nueva_resenia = [ps.stem(word) for word in nueva_resenia if not word in set(all_stopwords)]
nueva_resenia = ' '.join(nueva_resenia)
# Obtención del nuevo corpus limpio sin acotaciones, signos especiales ni artículos
corpus_nuevo = [nueva_resenia]
# Predicciones Nuevas
nuevo_X_test = cv.transform(corpus_nuevo).toarray()
nuevo_y_pred = NB_clasificador.predict(nuevo_X_test)
print(nuevo_y_pred)
