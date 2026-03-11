import sqlite3

DB_PATH = "wellness.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS encuestas (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES usuarios(id),
            datos_json  TEXT NOT NULL,
            pred_fisico REAL,
            pred_mental REAL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            encuesta_id INTEGER NOT NULL REFERENCES encuestas(id),
            user_id     INTEGER NOT NULL REFERENCES usuarios(id),
            real_fisico REAL NOT NULL,
            real_mental REAL NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS perfil_usuario (
            user_id    INTEGER PRIMARY KEY REFERENCES usuarios(id),
            edad       REAL NOT NULL,
            sexo       REAL NOT NULL,
            altura     REAL NOT NULL,
            peso       REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS metricas_modelo (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL REFERENCES usuarios(id),
            n_feedbacks      INTEGER NOT NULL,
            mae_fisico       REAL NOT NULL,
            mae_mental       REAL NOT NULL,
            error_medio      REAL NOT NULL,
            mejora_pct       REAL,
            pred_base_fisico REAL,
            pred_base_mental REAL,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()
    print("Base de datos inicializada correctamente.")