from enum import IntEnum
import numpy as np
from numpy.typing import NDArray
import unicodedata
import re
import json
from pathlib import Path
from datetime import date

VectorMusculo = NDArray[np.float32]


class Musculo(IntEnum):
    # Pecho
    PECTORAL                  = 0
    # Hombros
    DELTOIDES_ANTERIOR        = 1
    DELTOIDES_LATERAL         = 2
    DELTOIDES_POSTERIOR       = 3
    # Espalda
    TRAPECIO                  = 4  
    ESPALDA_ALTA              = 5
    ESPALDA_BAJA              = 6
    # Brazo
    BICEPS                    = 7
    TRICEPS                   = 8
    ANTEBRAZO                 = 9
    # Core
    ABDOMEN                   = 10
    # Pierna
    CUADRICEPS                = 11
    ISQUIOTIBIALES            = 12
    GLUTEOS                   = 13
    GEMELO                    = 14
    SOLEO                     = 15

cuentaMusculo = len(Musculo)

def make_muscle_vector(values: dict[Musculo, float]) -> VectorMusculo:
    # vector muscular saliente de diccionario, músculos no inicializados se quedan en 0.0
    vec = np.zeros(cuentaMusculo, dtype=np.float32)
    for musculo, nivel in values.items():
        if not 0.0 <= nivel <= 10.0:
            raise ValueError(f"{musculo.name}: nivel {nivel} fuera de rango")
        vec[musculo] = nivel 
    return vec

class User:
    def __init__(self, edad: int, experiencia: int, masa_magra: float, vector_musculo: VectorMusculo):
        self.edad = edad
        self.experiencia = experiencia
        self.masa_magra = masa_magra
        self.vector_musculo = vector_musculo

# rango útil frente al vector [0-10]
ESCALA_GLOBAL = 0.05
 
# lexico carga
def cargar_lexico(ruta: str | Path) -> dict[str, VectorMusculo]:
    """
    Lee lexico_ejercicios.json y devuelve:
        { nombre_normalizado → array float32[16] con activación 0–10 }
    """
    with open(ruta, encoding="utf-8") as f:
        raw: dict = json.load(f)
 
    lexico = {}
    for nombre, activaciones in raw.items():
        vector = np.zeros(cuentaMusculo, dtype=np.float32)
        for musculo, valor in activaciones.items():
            try:
                idx = Musculo[musculo]
            except KeyError:
                raise ValueError(f"Músculo desconocido '{musculo}' en '{nombre}'")
            vector[idx] = float(valor)
        lexico[_normalizar(nombre)] = vector
 
    return lexico
 
def _normalizar(texto: str) -> str:
    """Minúsculas, sin acentos, espacios simples."""
    sin_tildes = unicodedata.normalize("NFD", texto)
    sin_tildes = "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", sin_tildes.strip().lower())
 
 
def buscar_ejercicio(nombre: str, lexico: dict[str, VectorMusculo]) -> VectorMusculo | None:
    """Búsqueda exacta primero, luego por substring. Devuelve None si no encuentra."""
    clave = _normalizar(nombre)
    if clave in lexico:
        return lexico[clave]
    for clave_lexico, vector in lexico.items():
        if clave in clave_lexico or clave_lexico in clave:
            return vector
    return None
 
# f_wear - cálculo de desgaste
# D = (I · vol · P_f) · M_ej
# I = intensidad (RIR) = 1 - RIR/10, vol = seriesxreps, P_f = esfuerzo percibido, M_ej = activación normalizada
def f_wear(
    series:    int,
    reps:      int,
    rir:       int | None,
    pf:        float,
    m_ej:      VectorMusculo,
) -> VectorMusculo:
    # vector de desgaste D
    I   = 1.0 - (rir / 10.0) if rir is not None else 1.0
    vol = series * reps 
    m_normalizado = m_ej / 10.0  # lleva M_ej al rango [0, 1]
 
    D = (I * vol * pf * ESCALA_GLOBAL) * m_normalizado
    return D.astype(np.float32)
 
# Aplica el desgaste D sobre la fatiga acumulada previa F_{t-1} y devuelve la nueva fatiga F_t.
def f_apply(
    f_prev:   VectorMusculo,
    d:        VectorMusculo,
    capacidad: VectorMusculo,
) -> tuple[VectorMusculo, list[str]]:
    # desgaste sobre fatiga acumulada previa
    # f_prev - fatiga acumulada antes de esta sesión, d - desgaste del entrenamiento actual, capacidad - vector de capacidad máxima del User
    f_nueva = f_prev + d
 
    alertas = [
        Musculo(i).name
        for i in range(cuentaMusculo)
        if (capacidad[i] - f_nueva[i]) < 0
    ]
 
    return f_nueva.astype(np.float32), alertas # fatiga act y musculos que superan capacidad max
 
 
def procesar_sesion(
    ejercicios:  list[dict],
    lexico:      dict[str, VectorMusculo],
    f_prev:      VectorMusculo,
    capacidad:   VectorMusculo,
    ruta_lexicon_personal: str | Path | None = None,  # nuevo
) -> tuple[VectorMusculo, list[str], list[str]]:
    d_total        = np.zeros(cuentaMusculo, dtype=np.float32)
    no_encontrados = []

    for ej in ejercicios:
        m_ej = buscar_ejercicio(ej["nombre"], lexico)
        if m_ej is None:
            no_encontrados.append(ej["nombre"])
            continue

        d = f_wear(
            series = ej["series"],
            reps   = ej["reps"],
            rir    = ej.get("rir"),
            pf     = ej.get("pf", 1.0),
            m_ej   = m_ej,
        )
        d_total += d

    f_nueva, alertas = f_apply(f_prev, d_total, capacidad)

    # Actualizar marcas personales si se proporcionó ruta
    if ruta_lexicon_personal is not None:
        actualizar_lexicon_personal(ejercicios, ruta_lexicon_personal)

    return f_nueva, alertas, no_encontrados
 

# lexicon personal pa guardar las marcas del user
def cargar_lexicon_personal(ruta: str | Path) -> dict:
    # carga el léxico personal, si no hay pos vacío
    ruta = Path(ruta)
    if not ruta.exists():
        return {}
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)
 
 
def guardar_lexicon_personal(lexicon: dict, ruta: str | Path) -> None:
    # guardarlo
    with open(ruta, encoding="utf-8", mode="w") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)
 
 
def _es_mejor_marca(nuevo: dict, actual: dict) -> bool:
    # true si el nuevo supera la marca que haya
    # si es BW, gana mayor volumen
    # con peso pues gana el mayor
    if nuevo["es_bw"]:
        return (nuevo["series"] * nuevo["reps"]) > (actual["series"] * actual["reps"])
 
    peso_nuevo   = nuevo.get("peso_kg") or 0.0
    peso_actual  = actual.get("peso_kg") or 0.0
 
    if peso_nuevo != peso_actual:
        return peso_nuevo > peso_actual
 
    # Mismo peso — desempate por volumen
    return (nuevo["series"] * nuevo["reps"]) > (actual["series"] * actual["reps"])
 
 
def actualizar_lexicon_personal(
    ejercicios: list[dict],
    ruta:       str | Path,
) -> dict:
    # compara ejercicio de sesión con marca registrada y actualiza
    lexicon = cargar_lexicon_personal(ruta)
 
    for ej in ejercicios:
        clave = _normalizar(ej["nombre"])
 
        nuevo = {
            "series":     ej["series"],
            "reps":       ej["reps"],
            "peso_kg":    ej.get("peso_kg"),
            "es_bw":      ej.get("es_bw", False),
            "updated_at": date.today().isoformat(),
        }
 
        if clave not in lexicon or _es_mejor_marca(nuevo, lexicon[clave]):
            lexicon[clave] = nuevo
 
    guardar_lexicon_personal(lexicon, ruta)
    return lexicon