import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('research_data/wellness_dataset.csv')

features = ['edad', 'sexo', 'altura', 'bmi', 'peso', 'h_sueno', 'c_sueno', 'n_estres', 'h_movil', 't_pantalla', 'pasos', 
     'i_ejercicio', 'm_ejercicio', 'alim_enteros', 'alim_procesados', 'g_proteina', 'g_fibra', 'h_social', 'h_sol',]

targets = ['bienestar_fisico', 'bienestar_mental']

x = df[features]
y = df[targets]

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2)

rf_regression = RandomForestRegressor(n_estimators=200, random_state=42)

# soporta regresión multi salida por defecto
rf_regression.fit(x_tr, y_tr)

y_pred = rf_regression.predict(x_te)

# evaluar targets por separado
mse_separado = mean_squared_error(y_te, y_pred, multioutput='raw_values')
rmse_separado = np.sqrt(mse_separado)
r2_separado = r2_score(y_te, y_pred, multioutput='raw_values')

print("--- Evaluación del Modelo ---")
for i, target in enumerate(targets):
    print(f"Target: {target}")
    print(f"  MSE:  {mse_separado[i]:.2f}")
    print(f"  RMSE: {rmse_separado[i]:.2f} (Error promedio en las unidades originales)")
    print(f"  R²:   {r2_separado[i]:.2f} (Varianza explicada, más cerca a 1.0 es mejor)\n")