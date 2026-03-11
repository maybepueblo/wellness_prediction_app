import json
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from database import init_db, get_connection
from auth import register_user, login_user, get_user_from_token
from modelo import predecir, reentrenar, FEATURES

# ── Inicialización ───────────────────────────────────────────
init_db()
app = FastAPI(title="Wellness Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Auth helper ──────────────────────────────────────────────
def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    token = auth.replace("Bearer ", "").strip()
    user = get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    return user

# ── Schemas ──────────────────────────────────────────────────
class AuthSchema(BaseModel):
    username: str
    password: str

class EncuestaSchema(BaseModel):
    edad:             float
    sexo:             float
    altura:           float
    bmi:              float
    peso:             float
    h_sueno:          float
    c_sueno:          float
    n_estres:         float
    h_movil:          float
    t_pantalla:       float
    pasos:            float
    i_ejercicio:      float
    m_ejercicio:      float
    alim_enteros:     float
    alim_procesados:  float
    g_proteina:       float
    g_fibra:          float
    h_social:         float
    h_sol:            float
    l_agua:           float
    n_alcohol:        float
    t_meditacion:     float

class EncuestaDiariaSchema(BaseModel):
    h_sueno:          float
    c_sueno:          float
    n_estres:         float
    h_movil:          float
    t_pantalla:       float
    pasos:            float
    i_ejercicio:      float
    m_ejercicio:      float
    alim_enteros:     float
    alim_procesados:  float
    g_proteina:       float
    g_fibra:          float
    h_social:         float
    h_sol:            float
    l_agua:           float
    n_alcohol:        float
    t_meditacion:     float

class FeedbackSchema(BaseModel):
    encuesta_id: int
    real_fisico: float
    real_mental: float

# ── Rutas frontend ───────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()

@app.get("/encuesta", response_class=HTMLResponse)
async def encuesta_page():
    with open("static/encuesta.html") as f:
        return f.read()

@app.get("/resultado", response_class=HTMLResponse)
async def resultado_page():
    with open("static/resultado.html") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    with open("static/dashboard.html") as f:
        return f.read()

# ── Auth endpoints ───────────────────────────────────────────
@app.post("/auth/register")
async def register(data: AuthSchema):
    if len(data.username) < 3:
        raise HTTPException(400, "El usuario debe tener al menos 3 caracteres")
    if len(data.password) < 6:
        raise HTTPException(400, "La contraseña debe tener al menos 6 caracteres")
    result = register_user(data.username, data.password)
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return result

@app.post("/auth/login")
async def login(data: AuthSchema):
    result = login_user(data.username, data.password)
    if not result["ok"]:
        raise HTTPException(401, result["error"])
    return result

# ── Perfil del usuario (datos estáticos) ────────────────────
@app.get("/perfil")
async def get_perfil(user: dict = Depends(get_current_user)):
    conn = get_connection()
    perfil = conn.execute(
        "SELECT edad, sexo, altura, peso FROM perfil_usuario WHERE user_id = ?",
        (user["user_id"],)
    ).fetchone()
    conn.close()
    if not perfil:
        return None
    return dict(perfil)

# ── Encuesta completa (primera vez) ─────────────────────────
@app.post("/encuesta")
async def submit_encuesta(
    data: EncuestaSchema,
    user: dict = Depends(get_current_user)
):
    datos = data.model_dump()
    # Calcular BMI actualizado por si acaso
    if datos["altura"] > 0:
        datos["bmi"] = round(datos["peso"] / (datos["altura"] / 100) ** 2, 2)

    prediccion = predecir(datos, user["user_id"])

    conn = get_connection()
    cursor = conn.cursor()

    # Guardar o actualizar perfil estático
    cursor.execute("""
        INSERT INTO perfil_usuario (user_id, edad, sexo, altura, peso)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            edad=excluded.edad, sexo=excluded.sexo,
            altura=excluded.altura, peso=excluded.peso,
            updated_at=CURRENT_TIMESTAMP
    """, (user["user_id"], datos["edad"], datos["sexo"], datos["altura"], datos["peso"]))

    cursor.execute("""
        INSERT INTO encuestas (user_id, datos_json, pred_fisico, pred_mental)
        VALUES (?, ?, ?, ?)
    """, (
        user["user_id"],
        json.dumps(datos),
        prediccion["bienestar_fisico"],
        prediccion["bienestar_mental"]
    ))
    conn.commit()
    encuesta_id = cursor.lastrowid
    conn.close()

    return {"encuesta_id": encuesta_id, "prediccion": prediccion}

# ── Encuesta diaria (siguientes veces) ──────────────────────
@app.post("/encuesta-diaria")
async def submit_encuesta_diaria(
    data: EncuestaDiariaSchema,
    user: dict = Depends(get_current_user)
):
    # Recuperar perfil estático del usuario
    conn = get_connection()
    perfil = conn.execute(
        "SELECT edad, sexo, altura, peso FROM perfil_usuario WHERE user_id = ?",
        (user["user_id"],)
    ).fetchone()
    conn.close()

    if not perfil:
        raise HTTPException(400, "No se encontró perfil. Completa primero la encuesta inicial.")

    # Combinar perfil estático + datos del día
    datos = dict(perfil)
    datos.update(data.model_dump())
    datos["bmi"] = round(datos["peso"] / (datos["altura"] / 100) ** 2, 2)

    prediccion = predecir(datos, user["user_id"])

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO encuestas (user_id, datos_json, pred_fisico, pred_mental)
        VALUES (?, ?, ?, ?)
    """, (
        user["user_id"],
        json.dumps(datos),
        prediccion["bienestar_fisico"],
        prediccion["bienestar_mental"]
    ))
    conn.commit()
    encuesta_id = cursor.lastrowid
    conn.close()

    return {"encuesta_id": encuesta_id, "prediccion": prediccion}

# ── Feedback y reentrenamiento ───────────────────────────────
@app.post("/feedback")
async def submit_feedback(
    data: FeedbackSchema,
    user: dict = Depends(get_current_user)
):
    conn = get_connection()
    encuesta = conn.execute(
        "SELECT id FROM encuestas WHERE id = ? AND user_id = ?",
        (data.encuesta_id, user["user_id"])
    ).fetchone()

    if not encuesta:
        conn.close()
        raise HTTPException(404, "Encuesta no encontrada")

    existing = conn.execute(
        "SELECT id FROM feedback WHERE encuesta_id = ? AND user_id = ?",
        (data.encuesta_id, user["user_id"])
    ).fetchone()

    if existing:
        conn.execute("""
            UPDATE feedback SET real_fisico = ?, real_mental = ?
            WHERE encuesta_id = ? AND user_id = ?
        """, (data.real_fisico, data.real_mental, data.encuesta_id, user["user_id"]))
    else:
        conn.execute("""
            INSERT INTO feedback (encuesta_id, user_id, real_fisico, real_mental)
            VALUES (?, ?, ?, ?)
        """, (data.encuesta_id, user["user_id"], data.real_fisico, data.real_mental))

    conn.commit()
    conn.close()

    reentrenar(user["user_id"])

    return {"ok": True, "mensaje": "Feedback guardado y modelo actualizado"}

# ── Historial del usuario ────────────────────────────────────
@app.get("/historial")
async def get_historial(user: dict = Depends(get_current_user)):
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            e.id, e.pred_fisico, e.pred_mental, e.created_at,
            f.real_fisico, f.real_mental
        FROM encuestas e
        LEFT JOIN feedback f ON f.encuesta_id = e.id AND f.user_id = e.user_id
        WHERE e.user_id = ?
        ORDER BY e.created_at DESC
    """, (user["user_id"],)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── Check primera vez ────────────────────────────────────────
@app.get("/check-primera-vez")
async def check_primera_vez(user: dict = Depends(get_current_user)):
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) as c FROM encuestas WHERE user_id = ?",
        (user["user_id"],)
    ).fetchone()["c"]
    conn.close()
    return {"primera_vez": count == 0}