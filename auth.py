from datetime import datetime, timedelta
from typing import Optional
import bcrypt
from jose import JWTError, jwt
from database import get_connection

SECRET_KEY = "wellness_secret_key_change_in_production"
ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 24

# ── Contraseñas ──────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))

# ── JWT ──────────────────────────────────────────────────────
def create_token(user_id: int, username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# ── Usuarios ─────────────────────────────────────────────────
def register_user(username: str, password: str) -> dict:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        existing = cursor.execute(
            "SELECT id FROM usuarios WHERE username = ?", (username,)
        ).fetchone()

        if existing:
            return {"ok": False, "error": "El usuario ya existe"}

        hashed = hash_password(password)
        cursor.execute(
            "INSERT INTO usuarios (username, password_hash) VALUES (?, ?)",
            (username, hashed)
        )
        conn.commit()
        user_id = cursor.lastrowid
        token = create_token(user_id, username)
        return {"ok": True, "token": token, "user_id": user_id, "username": username}
    finally:
        conn.close()

def login_user(username: str, password: str) -> dict:
    conn = get_connection()
    try:
        user = conn.execute(
            "SELECT * FROM usuarios WHERE username = ?", (username,)
        ).fetchone()

        if not user or not verify_password(password, user["password_hash"]):
            return {"ok": False, "error": "Credenciales incorrectas"}

        token = create_token(user["id"], user["username"])
        return {"ok": True, "token": token, "user_id": user["id"], "username": user["username"]}
    finally:
        conn.close()

def get_user_from_token(token: str) -> Optional[dict]:
    payload = decode_token(token)
    if not payload:
        return None
    return {"user_id": int(payload["sub"]), "username": payload["username"]}