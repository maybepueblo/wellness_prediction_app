import numpy as np
from pathlib import Path

from parser import parsear_entrenamiento
from esfuerzo_notas          import cargar_analizador, analizar_nota
from metodos_musculos        import (
    cargar_lexico, procesar_sesion,
    Musculo, cuentaMusculo, actualizar_lexicon_personal,
)

# simulación de entrada
TEXTO_SESION = """
Entrenamiento hoy
* Press banca 3x12 50kg 3RIR , Me costó más de lo normal
* Jalón de triceps 4x12 25kg 5RIR , He mejorado bastante
* Flexiones 4x12 BW , No sentí nada raro, serie completada
* Polea de triceps 3x12 25kg , Molestias en el brazo
"""

RUTA_LEXICO = Path("lexico_ejercicios.json")

# estado inicial del User
f_prev    = np.zeros(cuentaMusculo, dtype=np.float32)   # sin fatiga previa
capacidad = np.full(cuentaMusculo, 10.0, dtype=np.float32)  # capacidad máxima


# parser
print("=" * 50)
print("Parsing.....")
print("=" * 50)

ejercicios_log = parsear_entrenamiento(TEXTO_SESION)

for ej in ejercicios_log:
    print(f"  {ej.nombre:<22} {ej.series}x{ej.reps}  "
          f"peso={'BW' if ej.es_bw else f'{ej.peso_kg}kg':<8} "
          f"RIR={ej.rir}  nota='{ej.nota_raw}'")


# NLP y modificador de esfuerzo percibido P_f
print()
print("=" * 50)
print("NLP (cargando modelo...)")
print("=" * 50)

analizador = cargar_analizador()

for ej in ejercicios_log:
    pf = analizar_nota(ej.nota_raw, analizador)
    print(f"  '{ej.nota_raw}'")
    print(f"    → P_f = {pf}")


# desgaste muscular y fatiga acumulada
print()
print("=" * 50)
print("PASO 3 — Cálculo de desgaste muscular")
print("=" * 50)

lexico = cargar_lexico(RUTA_LEXICO)

# Construir la lista de dicts que espera procesar_sesion
sesion = [
    {
        "nombre": ej.nombre,
        "series": ej.series,
        "reps":   ej.reps,
        "rir":    ej.rir,
        "pf":     analizar_nota(ej.nota_raw, analizador),
        "peso_kg": ej.peso_kg,   # nuevo
        "es_bw":   ej.es_bw, 
    }
    for ej in ejercicios_log
]

f_nueva, alertas, no_encontrados = procesar_sesion(
    sesion, lexico, f_prev, capacidad, Path("lexicon_personal.json")
)

print()
print("Desgaste por músculo:")
for musculo in Musculo:
    valor = f_nueva[musculo]
    if valor > 0:
        barra = "█" * int(valor * 200)
        print(f"  {musculo.name:<22}  {valor:.4f}")

print()
if alertas:
    print(f"ALERTA — músculos que superan capacidad: {alertas}")
else:
    print("Sin alertas — todos los músculos dentro de capacidad.")
    

if no_encontrados:
    print(f"Ejercicios no encontrados en léxico: {no_encontrados}")
