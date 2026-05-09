# ================================================================
# 03_audio_nested_loso.py ? AUDIO (HuBERT + MLP) NESTED LOSO
#   PREREQUISITO: ejecutar 00_config.py
#   SALIDA: nested_audio_logits.pkl
#
#   LÓGICA (sin fuga de datos):
#     Para cada fold externo K (Session K = test de fusión):
#       Para cada sesión J:
#         J == K ? MLP entrenado en {1..5} \ {K}      ? infer en K
#         J != K ? MLP entrenado en {1..5} \ {K, J}   ? infer en J
#
#   CACHÉ: 15 entrenamientos únicos máximo.
# ================================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

import torchaudio
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

MODELO_HUBERT = "facebook/hubert-large-ll60k"

RUTA_EMBEDS = os.path.join(RUTA_NESTED, "audio_embeddings")
RUTA_CKPTS  = os.path.join(RUTA_NESTED, "audio_ckpts")

# Mapeo incluyendo exc?hap para el parser de audio
EMOCION_A_CLASE = {
    'ang': 'anger', 'hap': 'happiness',
    'exc': 'happiness', 'sad': 'sadness', 'neu': 'neutral',
}

# ================================================================
# ?? PARSER AUDIO
# ================================================================

def _leer_emociones_audio(session_path: str) -> dict:
    """Igual que leer_emociones pero mapea exc?hap directamente."""
    import re
    emociones = {}
    emo_dir = os.path.join(session_path, "dialog", "EmoEvaluation")
    patron  = re.compile(
        r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+([a-zA-Z]+)")
    for fn in sorted(os.listdir(emo_dir)):
        if not fn.endswith('.txt') or not es_real(fn): continue
        with open(os.path.join(emo_dir, fn), encoding='utf-8', errors='ignore') as f:
            for linea in f:
                m = patron.match(linea.strip())
                if m:
                    clase = EMOCION_A_CLASE.get(m.group(4).lower())
                    if clase:
                        emociones[m.group(3)] = {
                            'emocion':          clase,
                            'emocion_original': m.group(4).lower(),
                            'start':            float(m.group(1)),
                            'end':              float(m.group(2)),
                        }
    return emociones


def _ruta_wav(session_path: str, uid: str):
    partes = uid.split('_')
    if len(partes) < 3: return None
    conv = '_'.join(partes[:-1])
    return os.path.join(session_path, "sentences", "wav", conv, f"{uid}.wav")


# ================================================================
# ?? EXTRACCIÓN HuBERT (cacheada por sesión)
# ================================================================

def _extraer_embedding_hubert(wav_path, extractor, modelo) -> np.ndarray:
    if not os.path.exists(wav_path) or not es_real(wav_path):
        return None
    try:
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != AUDIO_SR:
            wav = torchaudio.transforms.Resample(sr, AUDIO_SR)(wav)
        audio = wav.squeeze(0).numpy()
        if len(audio) < 400:
            return None
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        inp = extractor(audio, sampling_rate=AUDIO_SR,
                        return_tensors="pt", padding=True
                        ).input_values.to(DEVICE)
        with torch.no_grad():
            out = modelo(inp, output_hidden_states=True)
            emb = torch.stack(out.hidden_states[-4:]).mean(0).mean(1).squeeze(0)
        return emb.cpu().numpy().astype(np.float32)
    except Exception:
        return None


def _extraer_embeddings_sesion(session_path, extractor, modelo_hub) -> dict:
    sn  = os.path.basename(session_path)
    pkl = os.path.join(RUTA_EMBEDS, f"{sn}_hubert.pkl")
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f: data = pickle.load(f)
        print(f"  ?? {sn}: {len(data)} embeddings cargados")
        return data

    emociones  = _leer_emociones_audio(session_path)
    embeddings = {}
    for uid, info in tqdm(emociones.items(), desc=f"    {sn} HuBERT"):
        wp = _ruta_wav(session_path, uid)
        if not wp or not os.path.exists(wp): continue
        emb = _extraer_embedding_hubert(wp, extractor, modelo_hub)
        if emb is not None:
            embeddings[uid] = {'embedding': emb, 'session': sn, **info}

    with open(pkl, 'wb') as f: pickle.dump(embeddings, f)
    print(f"  ?? {sn}: {len(embeddings)} embeddings guardados")
    return embeddings


# ================================================================
# ?? MLP CLASIFICADOR
# ================================================================

class EmotionMLP(nn.Module):
    def __init__(self, input_dim=1024, hdims=(512, 256, 128),
                 n_cls=NUM_CLASSES, drop=0.4):
        super().__init__()
        layers, prev = [], input_dim
        for h in hdims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(drop)]
            prev = h
        layers.append(nn.Linear(prev, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class _EmbDataset(Dataset):
    def __init__(self, data: dict):
        self.s = [
            (torch.tensor(v['embedding'], dtype=torch.float32),
             torch.tensor(EMO_TO_IDX[v['emocion']], dtype=torch.long))
            for v in data.values() if v['emocion'] in EMO_TO_IDX
        ]

    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i]


def _entrenar_mlp(train_data: dict, input_dim: int, epochs=60) -> EmotionMLP:
    ds     = _EmbDataset(train_data)
    if not ds: return None
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2,
                        drop_last=len(ds) > 64)
    model  = EmotionMLP(input_dim).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit   = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for emb, lbl in loader:
            emb, lbl = emb.to(DEVICE), lbl.to(DEVICE)
            opt.zero_grad(); crit(model(emb), lbl).backward(); opt.step()
        sched.step()

    model.eval()
    return model


def _inferir(model: EmotionMLP, data: dict) -> dict:
    """Devuelve {uid: {logits(4), probs(4), emocion, emocion_original}}."""
    items = [(uid, v) for uid, v in data.items()
             if v['emocion'] in EMO_TO_IDX]
    if not items or model is None:
        return {}

    embs_t = torch.tensor(
        np.stack([v['embedding'] for _, v in items]), dtype=torch.float32
    )
    with torch.no_grad():
        logits_all = model(embs_t.to(DEVICE)).cpu()
        probs_all  = F.softmax(logits_all, dim=-1)

    return {
        uid: {
            'logits':           logits_all[i].numpy().astype(np.float32),
            'probs':            probs_all[i].numpy().astype(np.float32),
            'emocion':          v['emocion'],
            'emocion_original': v['emocion_original'],
            'session':          v.get('session', ''),
        }
        for i, (uid, v) in enumerate(items)
    }


# ================================================================
# ?? NESTED LOSO
# ================================================================

def nested_loso_audio() -> dict:
    # ?? 1. Embeddings HuBERT ????????????????????????????????????
    print("\n?? Cargando HuBERT...")
    extractor  = Wav2Vec2FeatureExtractor.from_pretrained(MODELO_HUBERT)
    modelo_hub = HubertModel.from_pretrained(MODELO_HUBERT).to(DEVICE)
    modelo_hub.eval()
    input_dim = modelo_hub.config.hidden_size   # 1024

    embeds = {}
    for sp in SESIONES:
        sn  = os.path.basename(sp)
        idx = int(sn.replace("Session", ""))
        emociones = _leer_emociones_audio(sp)
        embs      = _extraer_embeddings_sesion(sp, extractor, modelo_hub)
        # Sincronizar etiquetas desde el parser
        for uid in list(embs.keys()):
            if uid in emociones:
                embs[uid].update(emociones[uid])
            else:
                del embs[uid]
        embeds[idx] = embs
        print(f"  {sn}: {len(embs)} utterances válidos")

    del modelo_hub, extractor
    limpiar_memoria("HuBERT descargado")

    # ?? 2. Nested LOSO ??????????????????????????????????????????
    mlp_cache  = {}    # frozenset(train_ids) ? modelo entrenado
    nested_out = {}    # nested_out[K][J] = {uid: {...}}

    for K in SESSION_IDS:
        nested_out[K] = {}
        print(f"\n{'='*55}")
        print(f"?? FOLD K={K}  (Session{K} = TEST de fusión)")
        print(f"{'='*55}")

        for J in SESSION_IDS:
            ckpt = os.path.join(RUTA_CKPTS, f"K{K}_J{J}.pkl")
            if os.path.exists(ckpt):
                with open(ckpt, 'rb') as f:
                    nested_out[K][J] = pickle.load(f)
                print(f"  ? K={K} J={J} ? {len(nested_out[K][J])} utterances (ckpt)")
                continue

            train_ids = ([s for s in SESSION_IDS if s != K]       if J == K
                         else [s for s in SESSION_IDS if s not in (K, J)])
            cache_key = frozenset(train_ids)

            if cache_key not in mlp_cache:
                print(f"  ?? Entrenando MLP | train={sorted(train_ids)} ...")
                train_data = {}
                for sid in train_ids:
                    train_data.update(embeds[sid])
                mlp_cache[cache_key] = _entrenar_mlp(train_data, input_dim)
                print(f"     ? MLP listo (cache={len(mlp_cache)})")
            else:
                print(f"  ??  MLP reutilizado | train={sorted(train_ids)}")

            results = _inferir(mlp_cache[cache_key], embeds[J])
            nested_out[K][J] = results
            with open(ckpt, 'wb') as f: pickle.dump(results, f)
            print(f"  ?? K={K} J={J} ? {len(results)} logits")

    # ?? 3. Guardar + métricas ???????????????????????????????????
    out_path = os.path.join(RUTA_NESTED, "nested_audio_logits.pkl")
    with open(out_path, 'wb') as f: pickle.dump(nested_out, f)
    print(f"\n? Guardado: {out_path}")

    print(f"\n{'='*55}")
    print("?? Accuracy por fold (Audio MLP ? diagnóstico LOSO)")
    print(f"{'='*55}")
    all_true, all_pred, accs = [], [], []
    for K in SESSION_IDS:
        y_t = [EMO_TO_IDX[v['emocion']] for v in nested_out[K][K].values()
               if v['emocion'] in EMO_TO_IDX]
        y_p = [int(np.argmax(v['logits'])) for v in nested_out[K][K].values()
               if v['emocion'] in EMO_TO_IDX]
        if y_t:
            acc = accuracy_score(y_t, y_p)
            accs.append(acc)
            all_true.extend(y_t); all_pred.extend(y_p)
            print(f"  Session{K}: {acc:.4f}  ({len(y_t)} utterances)")

    if accs:
        print(f"\n  LOSO promedio : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        report_accuracy(
            [IDX_TO_EMO[i] for i in all_true],
            [IDX_TO_EMO[i] for i in all_pred],
            titulo="Audio HuBERT+MLP ? GLOBAL",
        )

    return nested_out


# ================================================================
# ?? MAIN
# ================================================================

if __name__ == "__main__":
    nested_audio = nested_loso_audio()
    print("\n? 03_audio_nested_loso completado")
    print(f"   Estructura: nested[K][J][uid] = {{logits, probs, emocion}}")
