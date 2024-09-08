
# Sleep Health Prediction Model

## Descripción

Este proyecto se enfoca en la creación y optimización de un modelo de regresión logística para predecir trastornos del sueño utilizando el conjunto de datos "Sleep Health". El objetivo es identificar los factores que afectan los trastornos del sueño y construir un modelo predictivo preciso que pueda generalizar bien a nuevos datos. Se utiliza un enfoque integral que incluye la limpieza de datos, ingeniería de características, y la optimización de modelos mediante técnicas avanzadas.

## Reporte del proyecto
Reporte Final: Archivo pdf con el reporte completo del proyecto.
Acceder al reporte [aquí](https://github.com/Maffergu/PortafolioFinalML/blob/main/Reporte-2.pdf)

## Conjunto de Datos

El conjunto de datos utilizado es el "Sleep Health Dataset" extraído de [Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset). Contiene información sobre diversos factores relacionados con la salud del sueño de los individuos, incluyendo variables como la duración del sueño, calidad del sueño, nivel de actividad física, nivel de estrés, índice de masa corporal (IMC), presión arterial, frecuencia cardíaca, y otros indicadores de salud.

## Metodología

1. **Limpieza de Datos**
   - Se corrigió la columna 'Sleep Disorder', transformando los valores nulos en 'No'.
   - Se creó una nueva columna binaria 'Has Sleep Disorder' para la variable objetivo.
   - Se aplicó `one-hot encoding` a las variables categóricas 'BMI Category' y 'Gender'.
   - Se eliminaron las columnas innecesarias, como 'Occupation'.

2. **Transformación de Datos**
   - La columna 'Blood Pressure' fue convertida al promedio de la presión sistólica y diastólica.
   - Se transformaron las variables categóricas utilizando `one-hot encoding`.

3. **Modelo de Decisión**
   - Se eligió el modelo de regresión logística para la predicción binaria.
   - Se implementaron técnicas de regularización y ajuste de parámetros utilizando `Grid Search` para encontrar los mejores hiperparámetros.

4. **Evaluación del Modelo**
   - Se dividió el conjunto de datos en entrenamiento, validación y prueba.
   - Se evaluó el modelo utilizando métricas de precisión y se generaron informes de clasificación para ambos conjuntos.

## Resultados

- **Mejores Parámetros:** {'C': 100, 'penalty': 'l1', 'solver': 'liblinear'}
- **Precisión en el Conjunto de Prueba:** 96.49%
- **Precisión en el Conjunto de Validación:** 94.64%

El modelo optimizado mostró una excelente capacidad de generalización, con alta precisión en los conjuntos de prueba y validación. Esto indica que el modelo es robusto y efectivo para predecir trastornos del sueño en nuevos datos.

## Cómo Ejecutar el Proyecto

1. **Instalar Dependencias**
   Asegúrate de tener las siguientes bibliotecas instaladas:
   ```bash
   pip install pandas scikit-learn graphviz
   ```

2. **Preparar el Conjunto de Datos**
   Si tienes problemas con el csv del repositorio, descarga el archivo `sleep_health.csv` del [dataset en Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) y colócalo en el directorio del proyecto.

3. **Ejecutar el Código**
   Ejecuta los scripts de Python para realizar el análisis y generar los resultados:
   ```bash
   python main.py
   ```

4. **Visualizar el Árbol de Decisión**
   El árbol de decisión se guardará como un archivo PDF llamado `decision_tree.pdf` en el directorio de trabajo.

## Archivos del Proyecto

- `main.py`: Script principal que realiza el análisis y optimización del modelo.
- `decTree.py`: Script con función para la creación de árbol de decisión.
- `sleep_health.csv`: Conjunto de datos utilizado para el análisis.
- `decision_tree.pdf`: Visualización del árbol de decisión generado.


## Contribuciones

Si deseas contribuir a este proyecto, por favor haz un fork del repositorio y envía un pull request con tus cambios.
