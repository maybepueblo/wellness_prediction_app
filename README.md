# 🌿 Wellness Prediction App

App web con login, encuestas, predicciones de bienestar y reentrenamiento personalizado por usuario.

## Estructura

```
wellness_prediction/
├── main.py              ← FastAPI backend (todos los endpoints)
├── database.py          ← SQLite: creación de tablas y conexión
├── auth.py              ← Registro, login, JWT
├── modelo.py            ← Predicción y reentrenamiento personalizado
├── requirements.txt
│
├── models/
│   └── base_wellness.pkl       ← NECESARIO: tu modelo entrenado con app.py
│
├── data/
│   └── wellness_dataset.csv    ← NECESARIO: tu dataset original
│
└── static/
    ├── index.html       ← Login / Registro
    ├── encuesta.html    ← Formulario de 22 variables
    ├── resultado.html   ← Predicción + feedback
    └── dashboard.html   ← Historial y gráfico de evolución
```

## Puesta en marcha

### 1. Instalar dependencias

```bash
uv pip install -r requirements.txt
```

### 2. Preparar el modelo base

Asegúrate de que existe `models/base_wellness.pkl`.
Si no existe, ejecútalo desde tu carpeta de entrenamiento:

```bash
python app.py
cp models/randomforest_wellness.pkl models/base_wellness.pkl
```

O añade al final de `app.py`:

```python
import shutil
shutil.copy('models/randomforest_wellness.pkl', 'models/base_wellness.pkl')
```

### 3. Arrancar el servidor

```bash
uv run uvicorn main:app --reload --port 8000
```

Abre el navegador en: **http://localhost:8000**

---

## Flujo de usuario

1. **Registro / Login** → `index.html`
2. **Primera vez** → redirige automáticamente a la encuesta
3. **Encuesta** → 22 variables con sliders e inputs
4. **Resultado** → predicción de bienestar físico y mental
5. **Feedback** → el usuario ajusta los valores reales → el modelo se reentrena
6. **Dashboard** → historial de todas las predicciones con gráfico de evolución

---

## Cómo funciona el reentrenamiento personalizado

Cada usuario tiene su propio modelo en `models/{user_id}_wellness.pkl`.

Cuando el usuario da feedback, se combinan:
- **Dataset global** (`wellness_dataset.csv`) con peso 1.0
- **Datos del usuario** (todos sus feedbacks) con peso 5.0

Así el modelo base da el comportamiento general, pero los datos propios
del usuario tienen 5x más influencia, personalizando la predicción con el tiempo.

Para cambiar el peso de los datos del usuario, edita `modelo.py`:
```python
USER_WEIGHT = 5.0  # aumentar = más personalizado, menor = más conservador
```

---

## Endpoints API

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/auth/register` | Crear cuenta |
| POST | `/auth/login` | Login, devuelve JWT |
| POST | `/encuesta` | Enviar datos, obtener predicción |
| POST | `/feedback` | Enviar valoración real, reentrena modelo |
| GET  | `/historial` | Historial del usuario autenticado |
| GET  | `/check-primera-vez` | Saber si el usuario tiene encuestas |
