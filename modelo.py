import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from database import get_connection

FEATURES = [
    'edad', 'sexo', 'altura', 'bmi', 'peso', 'h_sueno', 'c_sueno',
    'n_estres', 'h_movil', 't_pantalla', 'pasos', 'i_ejercicio',
    'm_ejercicio', 'alim_enteros', 'alim_procesados', 'g_proteina',
    'g_fibra', 'h_social', 'h_sol', 'l_agua', 'n_alcohol', 't_meditacion'
]
TARGETS    = ['bienestar_fisico', 'bienestar_mental']
BASE_MODEL = "models/base_wellness_model.pkl"
MODELS_DIR = "models"
USER_WEIGHT = 30.0

os.makedirs(MODELS_DIR, exist_ok=True)


def _load_model(user_id: int) -> Pipeline:
    user_path = f"{MODELS_DIR}/{user_id}_wellness.pkl"
    path = user_path if os.path.exists(user_path) else BASE_MODEL
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el modelo base en '{BASE_MODEL}'. "
            "Entrena primero el modelo con app.py."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["pipeline"]


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestRegressor(n_estimators=200, random_state=42))
    ])


def predecir(datos: dict, user_id: int) -> dict:
    pipeline = _load_model(user_id)
    X = pd.DataFrame([datos])[FEATURES]
    pred = pipeline.predict(X)[0]
    return {
        "bienestar_fisico": round(float(pred[0]), 1),
        "bienestar_mental":  round(float(pred[1]), 1),
    }


def reentrenar(user_id: int) -> dict:
    """
    Reentrena el modelo personalizado del usuario y devuelve métricas
    detalladas sobre la mejora conseguida respecto al modelo anterior.
    """
    conn = get_connection()

    # 1. Dataset global
    try:
        df_global = pd.read_csv("data/wellness_dataset.csv")
        X_global  = df_global[FEATURES].values
        y_global  = df_global[TARGETS].values
        w_global  = np.ones(len(X_global))
    except FileNotFoundError:
        X_global = np.empty((0, len(FEATURES)))
        y_global = np.empty((0, 2))
        w_global = np.array([])

    # 2. Feedback del usuario
    rows = conn.execute("""
        SELECT e.datos_json, f.real_fisico, f.real_mental, f.created_at
        FROM feedback f
        JOIN encuestas e ON f.encuesta_id = e.id
        WHERE f.user_id = ?
        ORDER BY f.created_at
    """, (user_id,)).fetchall()
    conn.close()

    if not rows:
        return {}

    X_user, y_user = [], []
    for row in rows:
        datos = json.loads(row["datos_json"])
        X_user.append([datos.get(f, 0) for f in FEATURES])
        y_user.append([row["real_fisico"], row["real_mental"]])

    X_user = np.array(X_user)
    y_user = np.array(y_user)
    w_user = np.full(len(X_user), USER_WEIGHT)

    # 3. Medir error del modelo ANTERIOR sobre los datos del usuario
    #    (cuánto se equivocaba antes de este reentrenamiento)
    try:
        old_pipeline = _load_model(user_id)
        y_pred_old   = old_pipeline.predict(X_user)
        mae_old_fis  = mean_absolute_error(y_user[:, 0], y_pred_old[:, 0])
        mae_old_men  = mean_absolute_error(y_user[:, 1], y_pred_old[:, 1])
        mae_old      = (mae_old_fis + mae_old_men) / 2
    except Exception:
        mae_old_fis = mae_old_men = mae_old = None

    # 4. Combinar y entrenar nuevo modelo
    if len(X_global) > 0:
        X_comb = np.vstack([X_global, X_user])
        y_comb = np.vstack([y_global, y_user])
        w_comb = np.concatenate([w_global, w_user])
    else:
        X_comb, y_comb, w_comb = X_user, y_user, w_user

    new_pipeline = _build_pipeline()
    new_pipeline.named_steps["imputer"].fit(X_comb)
    X_imp = new_pipeline.named_steps["imputer"].transform(X_comb)
    new_pipeline.named_steps["scaler"].fit(X_imp)
    X_scaled = new_pipeline.named_steps["scaler"].transform(X_imp)
    new_pipeline.named_steps["model"].fit(X_scaled, y_comb, sample_weight=w_comb)

    # 5. Medir error del modelo NUEVO sobre los mismos datos del usuario
    X_user_imp    = new_pipeline.named_steps["imputer"].transform(X_user)
    X_user_scaled = new_pipeline.named_steps["scaler"].transform(X_user_imp)
    y_pred_new    = new_pipeline.named_steps["model"].predict(X_user_scaled)
    mae_new_fis   = mean_absolute_error(y_user[:, 0], y_pred_new[:, 0])
    mae_new_men   = mean_absolute_error(y_user[:, 1], y_pred_new[:, 1])
    mae_new       = (mae_new_fis + mae_new_men) / 2

    # 6. Calcular mejora porcentual
    if mae_old and mae_old > 0:
        mejora_pct = round((mae_old - mae_new) / mae_old * 100, 1)
    else:
        mejora_pct = None

    # 7. Predicción del modelo base (sin personalizar) para comparar
    try:
        with open(BASE_MODEL, "rb") as f:
            base_bundle = pickle.load(f)
        base_pred    = base_bundle["pipeline"].predict(X_user)
        base_mae_fis = mean_absolute_error(y_user[:, 0], base_pred[:, 0])
        base_mae_men = mean_absolute_error(y_user[:, 1], base_pred[:, 1])
    except Exception:
        base_mae_fis = base_mae_men = None

    # 8. Guardar modelo personalizado
    bundle = {"pipeline": new_pipeline, "features": FEATURES, "targets": TARGETS}
    with open(f"{MODELS_DIR}/{user_id}_wellness.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # 9. Guardar métricas en BD
    conn = get_connection()
    conn.execute("""
        INSERT INTO metricas_modelo
            (user_id, n_feedbacks, mae_fisico, mae_mental, error_medio,
             mejora_pct, pred_base_fisico, pred_base_mental)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, len(rows),
        round(mae_new_fis, 3), round(mae_new_men, 3), round(mae_new, 3),
        mejora_pct, base_mae_fis, base_mae_men
    ))
    conn.commit()
    conn.close()

    metricas = {
        "n_feedbacks":   len(rows),
        "mae_fisico":    round(mae_new_fis, 2),
        "mae_mental":    round(mae_new_men, 2),
        "error_medio":   round(mae_new, 2),
        "mejora_pct":    mejora_pct,
        "mae_old":       round(mae_old, 2) if mae_old else None,
        "base_mae_fis":  round(base_mae_fis, 2) if base_mae_fis else None,
        "base_mae_men":  round(base_mae_men, 2) if base_mae_men else None,
        # Predicciones individuales para graficar pred vs real
        "detalle": [
            {
                "idx":        i + 1,
                "real_fis":   round(float(y_user[i, 0]), 1),
                "real_men":   round(float(y_user[i, 1]), 1),
                "pred_fis":   round(float(y_pred_new[i, 0]), 1),
                "pred_men":   round(float(y_pred_new[i, 1]), 1),
                "error_fis":  round(abs(float(y_user[i, 0]) - float(y_pred_new[i, 0])), 1),
                "error_men":  round(abs(float(y_user[i, 1]) - float(y_pred_new[i, 1])), 1),
            }
            for i in range(len(X_user))
        ]
    }

    print(f"[modelo] Usuario {user_id} | feedbacks={len(rows)} | "
          f"MAE={mae_new:.2f} | mejora={mejora_pct}%")
    return metricas