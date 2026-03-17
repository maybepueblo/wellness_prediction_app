import re
from dataclasses import dataclass

# salida
@dataclass
class EjercicioLog:
    nombre:   str
    series:   int
    reps:     int
    peso_kg:  float | None  # None si es bodyweight
    es_bw:    bool
    rir:      int | None    # None si no se especificó
    nota_raw: str           # NLP


# Ejemplo a capturar:
#   * Press banca 3x12 50kg 3RIR , Me dolía el hombro derecho
#   * Flexiones 4x12 BW (bodyweight) , Muy contento con este ejercicio
#   * Polea de triceps 3x12 25kg , Molestias en el brazo
_RE_LINEA = re.compile(
    r"^\*\s+"                               # bullet *
    r"(?P<nombre>.+?)\s+"                   # nombre del ejercicio (lazy)
    r"(?P<series>\d+)[xX](?P<reps>\d+)"     # NxM
    r"(?:"                                  # peso o BW (opcional)
        r"\s+(?P<peso>[\d.]+)\s*kg"         #   Xkg
        r"|\s+(?P<bw>BW\b[^,]*)"           #   BW (bodyweight)
    r")?"
    r"(?:\s+(?P<rir>\d+)\s*RIR)?"          # RIR opcional
    r"(?:\s*,\s*(?P<nota>.+))?$",           # , nota (opcional)
    re.IGNORECASE
)


def parsear_entrenamiento(texto: str) -> list[EjercicioLog]:
    # lineas sin * se ignoran, peso o BW opcionales, RIR opcional, nota opcional
    ejercicios = []

    for linea in texto.splitlines():
        linea = linea.strip()
        m = _RE_LINEA.match(linea)
        if not m:
            continue

        peso_kg: float | None = None
        es_bw = False

        if m.group("bw") is not None:
            es_bw = True
        elif m.group("peso") is not None:
            peso_kg = float(m.group("peso"))

        rir = int(m.group("rir")) if m.group("rir") is not None else None

        ejercicios.append(EjercicioLog(
            nombre   = m.group("nombre").strip(),
            series   = int(m.group("series")),
            reps     = int(m.group("reps")),
            peso_kg  = peso_kg,
            es_bw    = es_bw,
            rir      = rir,
            nota_raw = (m.group("nota") or "").strip(),
        ))

    return ejercicios