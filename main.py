import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from decTree import process_blood_pressure, decTree

# Cargar el archivo CSV
df = pd.read_csv('sleep_health.csv')

# Rellenar valores faltantes en 'Sleep Disorder' y crear variable objetivo binaria
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('No')  # Reemplaza valores nulos por 'No'
df['Has Sleep Disorder'] = df['Sleep Disorder'].apply(lambda x: 0 if x == 'No' else 1)  # Convertir a binario

# Eliminar la columna 'Sleep Disorder' ya que es redundante
df = df.drop('Sleep Disorder', axis=1)

# Codificar variables categóricas con one-hot encoding
df = pd.get_dummies(df, columns=['BMI Category', 'Gender'])

# Procesar la columna 'Blood Pressure' (presión arterial) usando la función personalizada
df = process_blood_pressure(df)  

# Eliminar columnas innecesarias si es necesario
df = df.drop('Occupation', axis=1)

# Guardar el DataFrame modificado en un archivo CSV
df.to_csv('modified_sleep_health.csv', index=False)

# Definir características (features) y objetivo (target)
X = df.drop('Has Sleep Disorder', axis=1)  # Características
y = df['Has Sleep Disorder']  # Objetivo

# Convertir todas las columnas de características a formato numérico
X = X.apply(pd.to_numeric, errors='coerce')

# Verificar si hay valores no numéricos restantes
print("Non-numeric values in features:", X.isna().sum())

# Manejar valores faltantes en las características, si los hay
X = X.fillna(0)  # Rellenar valores nulos con 0

# Definir proporciones para la división de datos
train_size = 0.7
test_size = 0.15
validation_size = 0.15

# Dividir en conjuntos de Entrenamiento + Validación y Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Ajustar proporción para la división de Validación
relative_validation_size = validation_size / (train_size + validation_size)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=relative_validation_size, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajustar y transformar datos de entrenamiento
X_test_scaled = scaler.transform(X_test)  # Transformar datos de prueba
X_validation_scaled = scaler.transform(X_validation)  # Transformar datos de validación

# Definir la cuadrícula de parámetros para GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Fuerza de regularización
    'penalty': ['l1', 'l2'],               # Tipo de regularización
    'solver': ['liblinear']                # Solver para optimización
}

# Inicializar GridSearchCV
grid_search = GridSearchCV(LogisticRegression(max_iter=2000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Ajustar GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Imprimir los mejores parámetros y la mejor puntuación
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Obtener el mejor modelo de la búsqueda de cuadrícula
best_log_reg = grid_search.best_estimator_

# Hacer predicciones en el conjunto de prueba
y_test_pred = best_log_reg.predict(X_test_scaled)
print("Accuracy Score on Test Set:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report on Test Set:\n", classification_report(y_test, y_test_pred))

# Hacer predicciones en el conjunto de validación
y_validation_pred = best_log_reg.predict(X_validation_scaled)

# Evaluar el modelo en el conjunto de validación
print("\nAccuracy Score on Validation Set:", accuracy_score(y_validation, y_validation_pred))
print("\nClassification Report on Validation Set:\n", classification_report(y_validation, y_validation_pred))

# Concatenar X_train con y_train y guardar en CSV
X_train_with_y = X_train.copy()
X_train_with_y['Has Sleep Disorder'] = y_train.values
X_train_with_y.to_csv('X_train_with_y_train.csv', index=False)

# Leer el archivo CSV guardado
df2 = pd.read_csv('X_train_with_y_train.csv')

# Asegurar que la función decTree pueda manejar el DataFrame con características y objetivo
decTree(df2)
