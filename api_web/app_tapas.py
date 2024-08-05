from flask import Flask, jsonify, request, render_template
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import pickle

root_path = '/home/emmaco/api_web/'

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), root_path, 'pagina_html'),
            static_folder=os.path.join(os.path.dirname(__file__), root_path, 'imagen'))

# Enruta la landing page (endpoint /)
@app.route("/")
def home():
    return render_template("Tapas.html")

# Enruta la función al endpoint /api/v1/predict
@app.route("/api/v1/predict", methods=["POST"])
def predict():
    data = request.json
    model_path = os.path.join(os.path.dirname(__file__), root_path, 'encurtidos.pkl')
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
        data_path = root_path + "datos/df_reentreno_desordenado.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)

            X = df[['Año de nacimiento', 'Comunidad', 'Género', 'Bebida']]
            y = df['Tapa']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Escalado de características
                ('clf', GradientBoostingClassifier(n_estimators=100,
                    learning_rate=0.2,
                    max_depth=7,
                    min_samples_split=5,
                    min_samples_leaf=1,
                    random_state=42))  # Clasificador
            ])

            # Entrenamiento del modelo
            pipeline.fit(X_train, y_train)

            # Evaluación del modelo
            accuracy = pipeline.score(X_test, y_test)

            # Guardar el mejor modelo entrenado
            with open(root_path + 'encurtidos.pkl', 'wb') as file:
                pickle.dump(pipeline, file)

            return f"Modelo reentrenado. Precisión: {accuracy:.4f}"
        else:
            return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>", 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
