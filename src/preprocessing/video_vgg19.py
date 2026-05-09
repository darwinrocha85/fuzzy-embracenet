# ================================================================
# 01_video_vgg19.py ? EXTRACCIÓN DE EMBEDDINGS (VGG19 + Haar)
#   PREREQUISITO: ejecutar 00_config.py
#   SALIDA: SessionX_embeddings_vgg19.pkl + metadata CSV
# ================================================================

# exec(open("00_config.py").read())  # ? descomentar si se ejecuta solo

import time
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from collections import defaultdict
from tqdm import tqdm

# ================================================================
# ?? MODELO
# ================================================================

class VGG19Emotion(nn.Module):
    """VGG19 (ImageNet) + Linear(512?4). Logits crudos."""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        self.features   = vgg19.features
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)          # [B, 4] logits


_transform_vgg19 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ================================================================
# ?? UTILIDADES PKL
# ================================================================

def _pkl_path(sn):  return os.path.join(RUTA_VIDEO_VGG, f"{sn}_embeddings_vgg19.pkl")
def _ckpt_path(sn): return os.path.join(RUTA_VIDEO_VGG, "checkpoints", f"{sn}_vgg19_ckpt.pkl")


def _cargar_embeddings(sn: str) -> dict:
    for path, key in [(_pkl_path(sn), None), (_ckpt_path(sn), 'embeddings')]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            data = data[key] if key else data
            print(f"    ?? [vgg19] {sn}: {len(data)} embeddings")
            return data
    return {}


def _guardar_checkpoint(embs: dict, sn: str):
    with open(_ckpt_path(sn), 'wb') as f:
        pickle.dump({'embeddings': embs}, f)


def _guardar_final(embs: dict, sn: str):
    with open(_pkl_path(sn), 'wb') as f:
        pickle.dump(embs, f)

    filas = []
    for uid, v in embs.items():
        fila = {
            'utterance_id':     uid,
            'emocion':          v['emocion'],
            'emocion_original': v['emocion_original'],
            'start':            v['timestamps'][0],
            'end':              v['timestamps'][1],
            'num_frames':       v['num_frames'],
            'face_detected':    v.get('face_detected', True),
        }
        for i, col in enumerate(PROB_COLS):
            fila[f'logit_{col}'] = float(v['logits'][i])
            fila[f'prob_{col}']  = float(v['probs'][i])
        filas.append(fila)

    csv_path = os.path.join(RUTA_VIDEO_VGG, f"{sn}_metadata_vgg19.csv")
    pd.DataFrame(filas).to_csv(csv_path, index=False)
    print(f"  ?? [vgg19] {sn}: {len(embs)} utterances ? {csv_path}")


# ================================================================
# ?? EXTRACCIÓN DE FRAMES
# ================================================================

def _extraer_frames_con_cara(video_path, start, end,
                              max_frames=VIDEO_FRAMES,
                              min_face=(30, 30)) -> tuple:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release(); return [], False

    sf    = int(start * fps)
    ef    = int(end   * fps)
    total = ef - sf
    if total <= 0:
        cap.release(); return [], False

    indices = (list(range(total)) if total <= max_frames
               else [int(i * total / max_frames) for i in range(max_frames)])

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf + idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=min_face)
        if len(faces) > 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if len(frames) > max_frames:
        idxs   = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idxs]

    return frames, bool(frames)


def extraer_embedding_vgg19(model, video_path, start, end) -> tuple:
    """
    Devuelve (logits_4, probs_4, n_frames, face_detected).
    Alineado con ViT FER: logits promediados ? softmax.
    """
    frames, face_det = _extraer_frames_con_cara(video_path, start, end)
    zeros = np.zeros(NUM_CLASSES, dtype=np.float32)

    if not face_det:
        return zeros, zeros.copy(), 0, False

    try:
        tensor = torch.stack([_transform_vgg19(f) for f in frames]).to(DEVICE)
    except Exception as e:
        print(f"    ??  Preprocesamiento: {e}")
        return zeros, zeros.copy(), 0, False

    with torch.no_grad():
        logits_all = model(tensor)          # [N, 4]

    logits_4 = logits_all.mean(0)           # [4]
    probs_4  = F.softmax(logits_4, 0)       # [4]

    return (
        logits_4.cpu().numpy().astype(np.float32),
        probs_4.cpu().numpy().astype(np.float32),
        len(frames),
        True,
    )


# ================================================================
# ?? PASO 1 ? Actualizar emociones en embeddings existentes
# ================================================================

def _actualizar_emociones():
    print("\n?? PASO 1: Actualizando emociones en embeddings existentes...")
    for sp in SESIONES:
        sn        = os.path.basename(sp)
        emociones = leer_emociones(sp)
        embs      = _cargar_embeddings(sn)
        if not embs:
            continue

        actualizados = eliminados = 0
        for uid in list(embs.keys()):
            if uid in emociones:
                nueva = emociones[uid]['emocion']
                if embs[uid].get('emocion') != nueva:
                    embs[uid]['emocion']          = nueva
                    embs[uid]['emocion_original'] = emociones[uid]['emocion_original']
                    actualizados += 1
            else:
                del embs[uid]; eliminados += 1

        if actualizados or eliminados:
            _guardar_final(embs, sn)
            print(f"  {sn} ? actualizados={actualizados}  eliminados={eliminados}")
        else:
            print(f"  {sn} ? emociones ya correctas")


# ================================================================
# ?? PASO 2 ? Extraer embeddings pendientes
# ================================================================

def _extraer_pendientes(model):
    print("\n??  PASO 2: Extrayendo embeddings VGG19...")
    t_global = time.time()

    for sp in SESIONES:
        sn        = os.path.basename(sp)
        emociones = leer_emociones(sp)
        gt_ids    = set(emociones)

        embs       = _cargar_embeddings(sn)
        pendientes = gt_ids - set(embs)

        if not pendientes:
            print(f"\n  {sn} ? ? Completo ({len(embs)} utterances)")
            continue
        print(f"\n  {sn} ? {len(pendientes)} pendientes de {len(gt_ids)}")

        avi_dir = os.path.join(sp, "dialog", "avi", "DivX")
        if not os.path.exists(avi_dir):
            print(f"  ? No encontrado: {avi_dir}"); continue

        utt_map = {uid: (v['start'], v['end']) for uid, v in emociones.items()}
        por_avi = defaultdict(list)
        for uid in pendientes:
            avi = nombre_avi(uid)
            if avi:
                por_avi[avi].append(uid)

        sin_archivo = sin_cara = 0
        t_sesion    = time.time()

        for avi_nombre, uids in tqdm(por_avi.items(), desc=f"    {sn}"):
            avi_path = os.path.join(avi_dir, avi_nombre)
            if not os.path.exists(avi_path) or not es_real(avi_nombre):
                sin_archivo += len(uids); continue

            for uid in uids:
                start, end = utt_map[uid]
                logits_4, probs_4, nf, face_det = extraer_embedding_vgg19(
                    model, avi_path, start, end)

                if not face_det:
                    sin_cara += 1

                info = emociones[uid]
                embs[uid] = {
                    'logits':           logits_4,
                    'probs':            probs_4,
                    'face_detected':    face_det,
                    'emocion':          info['emocion'],
                    'emocion_original': info['emocion_original'],
                    'vad':              info['vad'],
                    'session':          sn,
                    'timestamps':       (start, end),
                    'num_frames':       nf,
                    'prob_cols':        PROB_COLS,
                }

            if len(embs) % CHECKPOINT_N == 0:
                _guardar_checkpoint(embs, sn)

        _guardar_final(embs, sn)
        mins = (time.time() - t_sesion) / 60
        print(f"    ??  {mins:.1f} min | Total: {(time.time()-t_global)/60:.1f} min")
        if sin_archivo: print(f"    ??  {sin_archivo} utterances sin .avi")
        if sin_cara:    print(f"    ??  {sin_cara}  utterances sin cara")


# ================================================================
# ?? REPORTE
# ================================================================

def _reporte():
    print(f"\n{'='*60}")
    print("?? REPORTE VGG19 ? por sesión")
    print(f"{'='*60}")

    all_true, all_pred, all_probs = [], [], []

    for sp in SESIONES:
        sn   = os.path.basename(sp)
        embs = _cargar_embeddings(sn)
        if not embs:
            print(f"\n  {sn}: sin datos"); continue

        y_true, y_pred = [], []
        for v in embs.values():
            tl = v.get('emocion')
            if tl not in EMOCIONES_VALIDAS: continue
            pl = IDX_TO_EMO[int(np.argmax(v['probs']))]
            y_true.append(tl); y_pred.append(pl)
            all_true.append(tl); all_pred.append(pl)
            all_probs.append(v['probs'])

        no_face = sum(1 for v in embs.values() if not v.get('face_detected', True))
        dist    = dict(Counter(v['emocion'] for v in embs.values()))
        acc     = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        print(f"\n  {sn} | {len(y_true)} utterances | acc={acc:.2%} | no_cara={no_face}")
        print(f"    Distribución: {dist}")

    report_accuracy(all_true, all_pred, all_probs, "VGG19 ? GLOBAL")

    # CSV global
    if all_probs:
        df = pd.DataFrame({
            'true': all_true, 'pred': all_pred,
            **{f'prob_{e}': [p[i] for p in all_probs]
               for i, e in enumerate(EMOCIONES_VALIDAS)}
        })
        path = os.path.join(RUTA_VIDEO_VGG, "predictions_vgg19.csv")
        df.to_csv(path, index=False)
        print(f"\n  ?? CSV global ? {path}")


# ================================================================
# ?? MAIN
# ================================================================

if __name__ == "__main__":
    print("\n?? Cargando VGG19...")
    modelo_vgg19 = VGG19Emotion().to(DEVICE)
    modelo_vgg19.eval()
    print(f"? VGG19 listo | device={DEVICE}")

    _actualizar_emociones()
    _extraer_pendientes(modelo_vgg19)

    # Descargar modelo
    modelo_vgg19.cpu()
    del modelo_vgg19
    limpiar_memoria("vgg19 completado")

    _reporte()
    print("\n? 01_video_vgg19 completado")
