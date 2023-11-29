import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Tendriamos que realizar una manera de decidir a que valor coinciden los generos, directores, anios, repartos...
#tambien podriamos agregar infinitos factores para incrementar la precision del algoritmo, este ejemplo es basico.

data = {
    'Género': [0, 1, 2,  0, 3, 1],
    'Director': [1, 2, 3, 1, 4, 5],
    'Año': [2010, 2015, 2012, 2015, 2008, 2010],
    'Reparto': [2, 4, 1, 2, 1, 4],
    'Te gustó': [1, 0, 1, 0, 1, 0]  # 1 si te gustó, 0 si no te gustó
}

df = pd.DataFrame(data)

# Separar los datos en características (X) y etiquetas (y)
X = df.drop('Te gustó', axis=1)
y = df['Te gustó']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de árbol de decisión
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Supongamos que estamos considerando una película con las siguientes características:
nueva_pelicula = pd.DataFrame({
    'Género': [1],
    'Director': [1],
    'Año': [2014],
    'Reparto': [1]
})

# Preprocesa las características de la nueva película
nueva_pelicula_encoded = pd.get_dummies(nueva_pelicula)

# Utiliza el modelo entrenado para hacer una predicción
prediccion = clf.predict(nueva_pelicula_encoded)

# Determina si es probable que te guste o no
if prediccion[0] == 1:
    print("Es probable que te guste esta película.")
else:
    print("Es probable que no te guste esta película.")