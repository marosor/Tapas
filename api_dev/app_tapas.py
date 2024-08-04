from flask import Flask, jsonify, request, render_template
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

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'pagina_html'), 
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'imagen'))

# Enruta la landing page (endpoint /)
@app.route("/")
def home():
    return render_template("Tapas.html")

# Enruta la función al endpoint /api/v1/predict
@app.route("/api/v1/predict", methods=["POST"])
def predict():
    data = request.json
    model_path = os.path.join(os.path.dirname(__file__),'..' ,'modelo_entrenado', 'encurtidos.pkl')
    model = pickle.load(open(model_path, 'rb'))

    anio = data.get('anio', 0)
    comunidad = data.get('comunidad', 0)
    genero = data.get('genero', 0)
    bebida = data.get('bebida', 0)

    if anio is None or comunidad is None or genero is None or bebida is None:
        return jsonify({"error": "Datos insuficientes para predecir."}), 400
    else:
        prediction = model.predict([[int(anio), int(comunidad), int(genero), int(bebida)]])
        return jsonify({'prediction': int(prediction[0])})

# Enruta la función al endpoint /api/v1/retrain
@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    try:
        data_path = "../datos/df_reentreno_desordenado.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)

            X = df[['Año de nacimiento', 'Comunidad', 'Género', 'Bebida']]
            y = df['Tapa']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Escalado de características
                ('clf', GradientBoostingClassifier(random_state=42))  # Clasificador
            ])

            param_grid = [
                {
                    'clf__n_estimators': [100],
                    'clf__learning_rate': [0.2],
                    'clf__max_depth': [7],
                    'clf__min_samples_split': [5],
                    'clf__min_samples_leaf': [1]
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
    app.run(debug=True)
