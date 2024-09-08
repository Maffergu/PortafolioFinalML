import pandas as pd
import graphviz
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

def process_blood_pressure(df):
    # Convertir 'Blood Pressure' a la media de la presión sistólica y diastólica
    df['Blood Pressure'] = df['Blood Pressure'].apply(
        lambda x: sum(map(float, x.split('/'))) / 2 if isinstance(x, str) else x
    )
    return df

def decTree(df_train):
    # Eliminar la columna 'Person ID' ya que no es necesaria para el modelo
    df_train = df_train.drop('Person ID', axis=1)
    
    # Definir las características (features) y la variable objetivo (target)
    x_train = df_train.drop(['Has Sleep Disorder'], axis=1)  # Eliminar solo la columna objetivo
    
    y_train = df_train['Has Sleep Disorder']

    # Configurar el modelo de Árbol de Decisión
    tree_clf = tree.DecisionTreeClassifier(max_depth=10)

    # Entrenar el modelo
    tree_clf.fit(x_train, y_train)

    # Exportar el árbol de decisión utilizando Graphviz
    dot_data = tree.export_graphviz(
        tree_clf, 
        out_file=None, 
        feature_names=list(x_train.columns), 
        class_names=['0', '1'],  # Asumiendo clases binarias; ajustar si es necesario
        filled=True, 
        rounded=True, 
        special_characters=True
    )  

    # Crear un objeto Graphviz Source
    graph = graphviz.Source(dot_data)  

    # Renderizar y guardar como un archivo PDF
    graph.render("decision_tree", format="pdf", cleanup=True)
