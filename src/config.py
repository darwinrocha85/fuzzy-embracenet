# ================================================================
# config.py — CONFIGURACIÓN Y UTILIDADES COMPARTIDAS
#   Todos los scripts del proyecto importan desde aquí.
# ================================================================

import os, re, gc, pickle, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import Counter

# ── RUTAS ───────────────────────────────────────────────────────
RUTA_BASE         = "/kaggle/working/embeddings"
RUTA_VIDEO_VGG    = os.path.join(RUTA_BASE, "video_vgg19")
RUTA_VIDEO_HSE    = os.path.join(RUTA_BASE, "video_hsemotion")
RUTA_NESTED       = "/kaggle/working/nested_loso"
RUTA_FUSION       = os.path.join(RUTA_NESTED, "fusion_results")

for p in [RUTA_VIDEO_VGG, RUTA_VIDEO_HSE,
          os.path.join(RUTA_VIDEO_VGG, "checkpoints"),
          os.path.join(RUTA_VIDEO_HSE, "checkpoints"),
          os.path.join(RUTA_NESTED, "audio_embeddings"),
          os.path.join(RUTA_NESTED, "audio_ckpts"),
          os.path.join(RUTA_NESTED, "text_ckpts"),
          os.path.join(RUTA_NESTED, "text_models"),
          RUTA_FUSION]:
    os.makedirs(p, exist_ok=True)

# ── DEVICE ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── EMOCIONES ───────────────────────────────────────────────────
EMOCIONES_VALIDAS = ['anger', 'happiness', 'sadness', 'neutral']
EMO_TO_IDX        = {e: i for i, e in enumerate(EMOCIONES_VALIDAS)}
IDX_TO_EMO        = {i: e for i, e in enumerate(EMOCIONES_VALIDAS)}
PROB_COLS         = EMOCIONES_VALIDAS

# Mapeo raw IEMOCAP → clases del proyecto
EMOCIONES_MAP = {
    'ang': 'anger',    'hap': 'happiness', 'exc': 'excitement',
    'sad': 'sadness',  'neu': 'neutral',   'fru': 'frustration',
    'fea': 'fear',     'sur': 'surprise',  'dis': 'disgust',
    'oth': 'other',    'xxx': 'other',
}
MAP_EMOCION = {'excitement': 'happiness'}   # exc → hap

# ── PARÁMETROS COMUNES ──────────────────────────────────────────
VIDEO_FRAMES  = 8
IMG_SIZE      = 48
NUM_CLASSES   = 4
CHECKPOINT_N  = 10
AUDIO_SR      = 16_000
N_SESSIONS    = 5

# ================================================================
# 🔧 UTILIDADES COMPARTIDAS
# ================================================================

def es_real(nombre: str) -> bool:
    """Filtra archivos macOS ocultos (._*)."""
    return not os.path.basename(nombre).startswith('._')


def normalizar_emocion(emo: str):
    """Aplica MAP_EMOCION; devuelve None si no es válida."""
    emo = MAP_EMOCION.get(emo, emo)
    return emo if emo in EMOCIONES_VALIDAS else None


def limpiar_memoria(etapa: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    libre = shutil.disk_usage("/kaggle/working").free / 1e9
    print(f"  🧹 [{etapa}] Espacio libre: {libre:.1f} GB")


def nombre_avi(utt_id: str):
    """Construye el nombre del .avi para una utterance de IEMOCAP."""
    partes = utt_id.split('_')
    if len(partes) < 3:
        return None
    actor    = partes[-1][0]
    ses_base = partes[0][:-1] + actor
    conv     = '_'.join(partes[1:-1])
    return f"{ses_base}_{conv}.avi"


# ── IEMOCAP ─────────────────────────────────────────────────────
def encontrar_iemocap() -> str:
    candidatos = [
        "/kaggle/input/IEMOCAP/IEMOCAP_full_release",
        "/kaggle/input/datasets/jiten597/iemocap/IEMOCAP_full_release",
    ]
    try:
        for entry in os.listdir("/kaggle/input"):
            full = os.path.join("/kaggle/input", entry)
            if os.path.isdir(full):
                candidatos += [full, os.path.join(full, "IEMOCAP_full_release")]
    except Exception:
        pass

    for c in candidatos:
        if os.path.exists(c):
            sesiones = [d for d in os.listdir(c) if d.startswith("Session")]
            if sesiones:
                print(f"  ✅ IEMOCAP: {c}  |  sesiones: {sorted(sesiones)}")
                return c

    raise FileNotFoundError(
        "IEMOCAP no encontrado.\n👉 Kaggle: Add data → busca 'iemocap'."
    )


IEMOCAP_ROOT = encontrar_iemocap()
SESIONES = sorted([
    os.path.join(IEMOCAP_ROOT, d)
    for d in os.listdir(IEMOCAP_ROOT)
    if d.startswith("Session") and os.path.isdir(os.path.join(IEMOCAP_ROOT, d))
])
SESSION_IDS = [int(os.path.basename(s).replace("Session", "")) for s in SESIONES]


# ── PARSER IEMOCAP ──────────────────────────────────────────────
_PATRON_EMO = re.compile(
    r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+"
    r"(\S+)\s+([a-zA-Z]+)\s+"
    r"\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]"
)

def leer_emociones(session_path: str) -> dict:
    """
    Lee EmoEvaluation y filtra las 4 emociones válidas.
    excitement → happiness; frustration, fear, etc. → ignorados.
    """
    emociones = {}
    emo_path  = os.path.join(session_path, "dialog", "EmoEvaluation")
    if not os.path.exists(emo_path):
        print(f"  ⚠️  EmoEvaluation no encontrado: {emo_path}")
        return emociones

    for archivo in sorted(os.listdir(emo_path)):
        ruta = os.path.join(emo_path, archivo)
        if not es_real(archivo) or not os.path.isfile(ruta):
            continue
        if not archivo.endswith('.txt'):
            continue
        with open(ruta, encoding='utf-8', errors='ignore') as f:
            for linea in f:
                m = _PATRON_EMO.match(linea.strip())
                if not m:
                    continue
                utt_id    = m.group(3)
                emo_raw   = m.group(4).lower()
                emo_mapped = EMOCIONES_MAP.get(emo_raw, emo_raw)
                emo_final  = normalizar_emocion(emo_mapped)
                if emo_final is None:
                    continue
                emociones[utt_id] = {
                    'emocion':          emo_final,
                    'emocion_original': emo_raw,
                    'vad':   (float(m.group(5)), float(m.group(6)), float(m.group(7))),
                    'start':  float(m.group(1)),
                    'end':    float(m.group(2)),
                }
    return emociones


# ── MÉTRICAS ────────────────────────────────────────────────────
def report_accuracy(all_true: list, all_pred: list,
                    all_probs: list = None, titulo: str = ""):
    """Imprime accuracy global + por clase + matriz de confusión."""
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)
    import warnings; warnings.filterwarnings('ignore')

    if not all_true:
        print("  ⚠️  Sin datos para evaluar"); return

    acc = accuracy_score(all_true, all_pred)
    print(f"\n{'='*60}")
    if titulo:
        print(f"  {titulo}")
    print(f"  Accuracy global      : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Utterances evaluadas : {len(all_true)}")

    print(f"\n  📋 Classification Report:")
    print(classification_report(
        all_true, all_pred,
        labels=EMOCIONES_VALIDAS,
        target_names=EMOCIONES_VALIDAS,
        digits=4,
    ))

    cm = confusion_matrix(all_true, all_pred, labels=EMOCIONES_VALIDAS)
    print(f"  📊 Matriz de confusión (filas=GT, cols=PRED):")
    header = "             " + "  ".join(f"{e[:5]:>7}" for e in EMOCIONES_VALIDAS)
    print(header)
    for i, emo in enumerate(EMOCIONES_VALIDAS):
        fila = "  ".join(f"{cm[i,j]:>7}" for j in range(len(EMOCIONES_VALIDAS)))
        print(f"  {emo:<12} {fila}")
    print(f"{'='*60}")


# ── PRINT BIENVENIDA ────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ✅ Config cargada | device={DEVICE}")
print(f"  Sesiones : {[os.path.basename(s) for s in SESIONES]}")
print(f"  Emociones: {EMOCIONES_VALIDAS}")
print(f"  Rutas    : VGG→{RUTA_VIDEO_VGG}")
print(f"           : Nested→{RUTA_NESTED}")
print(f"{'='*60}")
