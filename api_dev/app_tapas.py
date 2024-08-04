from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle

app = Flask(__name__)
app.config['DEBUG'] = True

# Enruta la landing page (endpoint /)
@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a la API de Tapas"

# Enruta la función al endpoint /api/v1/predict
@app.route("/api/v1/predict", methods=["GET"])
def predict():
    try:
        # Cargar el modelo entrenado
        model_path = '../modelo_entrenado/encurtidos.pkl'
        if not os.path.exists(model_path):
            return jsonify({'error': 'Modelo no encontrado.'}), 404

        model = pickle.load(open(model_path, 'rb'))

        # Obtener los parámetros de la solicitud
        anio = request.args.get('anio')
        comunidad = request.args.get('comunidad')
        genero = request.args.get('genero')
        bebida = request.args.get('bebida')

        # Validar que los parámetros no estén vacíos
        if not all([anio, comunidad, genero, bebida]):
            return jsonify({'error': 'Faltan parámetros para la predicción.'}), 400

        # Convertir los parámetros a tipo int
        try:
            anio = int(anio)
            comunidad = int(comunidad)
            genero = int(genero)
            bebida = int(bebida)
        except ValueError:
            return jsonify({'error': 'Los parámetros deben ser números enteros.'}), 400

        # Realizar la predicción
        prediction = model.predict([[anio, comunidad, genero, bebida]])
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Enruta la función al endpoint /api/v1/retrain
@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    try:
        data_path = "../datos/df_desordenado_new.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)

            X = df[['Año de nacimiento', 'Comunidad', 'Género', 'Bebida']]
            y = df['Tapa']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Escalado de características
                ('clf', RandomForestClassifier(random_state=42))  # Clasificador
            ])

            param_grid = [
                {
                    'clf': [RandomForestClassifier(random_state=42)],
                    'clf__n_estimators': [50, 100, 200],
                    'clf__max_depth': [None, 10, 20, 30],
                    'clf__min_samples_split': [2, 5, 10],
                    'clf__min_samples_leaf': [1, 2, 4]
                },
                {
                    'clf': [GradientBoostingClassifier(random_state=42)],
                    'clf__n_estimators': [50, 100, 200],
                    'clf__learning_rate': [0.01, 0.1, 0.2],
                    'clf__max_depth': [3, 5, 7],
                    'clf__min_samples_split': [2, 5, 10],
                    'clf__min_samples_leaf': [1, 2, 4]
                },
                {
                    'clf': [SVC()],
                    'clf__C': [0.1, 1, 10],
                    'clf__kernel': ['linear', 'rbf'],
                    'clf__gamma': ['scale', 'auto']
                },
                {
                    'clf': [KNeighborsClassifier()],
                    'clf__n_neighbors': [3, 5, 7],
                    'clf__weights': ['uniform', 'distance']
                },
                {
                    'clf': [LogisticRegression(max_iter=10000)],
                    'clf__C': [0.1, 1, 10],
                    'clf__solver': ['lbfgs', 'liblinear']
                },
                {
                    'clf': [XGBClassifier(eval_metric='mlogloss')],
                    'clf__n_estimators': [50, 100, 200],
                    'clf__learning_rate': [0.01, 0.1, 0.2],
                    'clf__max_depth': [3, 5, 7],
                    'clf__subsample': [0.8, 1.0]
                },
                {
                    'clf': [LGBMClassifier()],
                    'clf__n_estimators': [50, 100, 200],
                    'clf__learning_rate': [0.01, 0.1, 0.2],
                    'clf__max_depth': [-1, 10, 20]
                }
            ]

            grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            accuracy = best_model.score(X_test, y_test)

            # Guardar el mejor modelo entrenado
            with open('../modelo_entrenado/encurtidos.pkl', 'wb') as file:
                pickle.dump(best_model, file)

            return f"Modelo reentrenado. Precisión: {accuracy:.4f}"
        else:
            return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>", 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run()
