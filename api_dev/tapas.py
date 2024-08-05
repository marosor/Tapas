import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import pickle

df = pd.read_csv("datos/df_desordenado.csv") 

X = df[['Año de nacimiento', 'Comunidad', 'Género', 'Bebida']]
y = df['Tapa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Escalado de características
    ('clf', GradientBoostingClassifier(random_state=42))  # Clasificador
])

param_grid = [
    {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__max_depth': [3, 5, 7],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    }
]

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

accuracy = best_model.score(X_test, y_test)

print(f"Mejores parámetros: {best_params}")
print(f"Rendimiento en el conjunto de prueba: {accuracy}")

with open('encurtidos.pkl', 'wb') as file:
    pickle.dump(best_model, file)