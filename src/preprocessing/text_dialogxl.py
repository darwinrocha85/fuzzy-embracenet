# ================================================================
# 04_text_nested_loso.py ? TEXTO (DialogXL) NESTED LOSO
#   PREREQUISITO: ejecutar 00_config.py
#   SALIDA: nested_text_logits.pkl
# ================================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

import random
import torch.nn as nn
from transformers import (XLNetTokenizer, XLNetModel,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import accuracy_score
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

MODELO_BASE   = "xlnet-base-cased"
MAX_TOKENS    = 2048
HIDDEN_SIZE   = 768
N_HEADS       = 8
N_LAYERS_ATTN = 2
DROPOUT       = 0.1
LR_BACKBONE   = 1e-5
LR_HEADS      = 5e-5
WEIGHT_DECAY  = 0.01
N_EPOCHS      = 10
WARMUP_RATIO  = 0.1
SEED          = 42

# 6 clases en train (inc. excitement y fear) ? fusión posterior a 4
EMOCIONES_6  = ['anger', 'happiness', 'sadness', 'neutral', 'excitement', 'fear']
EMO6_TO_IDX  = {e: i for i, e in enumerate(EMOCIONES_6)}

MAP_GT = {
    'ang': 'anger',   'hap': 'happiness', 'sad': 'sadness',
    'neu': 'neutral', 'exc': 'excitement', 'fea': 'fear',
    'fru': None, 'xxx': None, 'oth': None, 'sur': None, 'dis': None,
}

RUTA_CKPTS  = os.path.join(RUTA_NESTED, "text_ckpts")
RUTA_MODELS = os.path.join(RUTA_NESTED, "text_models")


def _set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

_set_seed()

# ================================================================
# ??? ARQUITECTURA
# ================================================================

class DialogXLClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained(MODELO_BASE)
        enc = nn.TransformerEncoderLayer(
            HIDDEN_SIZE, N_HEADS, HIDDEN_SIZE * 4, DROPOUT, batch_first=True)
        self.dialog_attn = nn.TransformerEncoder(enc, N_LAYERS_ATTN)
        self.gru = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE // 2,
                          bidirectional=True, batch_first=True)
        self.cls = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2),
            nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE // 2, len(EMOCIONES_6)),
        )

    def forward(self, input_ids, attention_mask, utt_spans):
        hidden = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        reps, valid_idx = [], []
        for i, span in enumerate(utt_spans):
            if span is None: continue
            s, e = span
            reps.append(hidden[0, s:e].mean(0))
            valid_idx.append(i)
        if not reps: return None, [], None
        x = torch.stack(reps).unsqueeze(0)
        x = self.dialog_attn(x)
        x, _ = self.gru(x)
        embs   = x.squeeze(0)
        logits = self.cls(embs)
        return logits, valid_idx, embs.detach()


# ================================================================
# ?? PARSER TEXTO
# ================================================================

def _leer_datos_sesion(path: str) -> list:
    import re
    emociones, trans = {}, {}
    emo_dir = os.path.join(path, "dialog", "EmoEvaluation")
    tra_dir = os.path.join(path, "dialog", "transcriptions")

    for fn in [f for f in os.listdir(emo_dir)
               if f.endswith('.txt') and not f.startswith('.')]:
        with open(os.path.join(emo_dir, fn)) as f:
            for line in f:
                m = re.match(
                    r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+([a-zA-Z]+)",
                    line.strip())
                if m:
                    emo = MAP_GT.get(m.group(4).lower())
                    if emo:
                        emociones[m.group(3)] = {
                            'emo': emo, 'start': float(m.group(1))}

    for fn in [f for f in os.listdir(tra_dir)
               if f.endswith('.txt') and not f.startswith('.')]:
        with open(os.path.join(tra_dir, fn)) as f:
            for line in f:
                m = re.match(r'^(\S+)\s+\[.+\]:\s*(.+)$', line.strip())
                if m and m.group(1) in emociones:
                    trans[m.group(1)] = m.group(2).strip()

    convs = defaultdict(list)
    for uid, info in emociones.items():
        if uid in trans:
            convs[uid.rsplit('_', 1)[0]].append({
                'uid':          uid,
                'texto':        trans[uid],
                'label':        EMO6_TO_IDX[info['emo']],
                'emocion_raw':  info['emo'],
                'start':        info['start'],
            })

    dataset = []
    for cid, utts in convs.items():
        utts.sort(key=lambda x: x['start'])
        dataset.append({
            'conv_id': cid,
            'utterances': utts,
            'session': os.path.basename(path),
        })
    return dataset


def _tokenizar_dialogo(textos: list, tokenizer) -> tuple:
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    tokens, spans = [], []
    for t in textos:
        ids = tokenizer.encode(t, add_special_tokens=False)
        avail = MAX_TOKENS - len(tokens) - 2
        if avail <= 0:
            spans.append(None); continue
        ids = ids[:avail]
        s = len(tokens)
        tokens.extend(ids)
        spans.append((s, len(tokens)))
        tokens.append(sep_id)
    tokens.append(cls_id)
    t = torch.tensor([tokens]).to(DEVICE)
    m = torch.ones(1, len(tokens)).to(DEVICE)
    return t, m, spans


# ================================================================
# ?? ENTRENAMIENTO Y CACHÉ
# ================================================================

def _model_path(train_ids: list) -> str:
    key = "_".join(str(s) for s in sorted(train_ids))
    return os.path.join(RUTA_MODELS, f"dialogxl_train{key}.pt")


def _entrenar_dialogxl(train_convs, tokenizer) -> DialogXLClassifier:
    modelo = DialogXLClassifier().to(DEVICE)
    opt = torch.optim.AdamW([
        {'params': modelo.xlnet.parameters(),        'lr': LR_BACKBONE},
        {'params': modelo.cls.parameters(),          'lr': LR_HEADS},
        {'params': modelo.dialog_attn.parameters(),  'lr': LR_HEADS},
        {'params': modelo.gru.parameters(),          'lr': LR_HEADS},
    ], weight_decay=WEIGHT_DECAY)
    total  = len(train_convs) * N_EPOCHS
    sched  = get_linear_schedule_with_warmup(
        opt, int(total * WARMUP_RATIO), total)
    crit   = nn.CrossEntropyLoss()

    for ep in range(1, N_EPOCHS + 1):
        modelo.train()
        random.shuffle(train_convs)
        t_loss = 0
        for conv in train_convs:
            ids, mask, spans = _tokenizar_dialogo(
                [u['texto'] for u in conv['utterances']], tokenizer)
            logits, v_idx, _ = modelo(ids, mask, spans)
            if logits is None: continue
            labels = torch.tensor(
                [conv['utterances'][i]['label'] for i in v_idx]).to(DEVICE)
            loss = crit(logits, labels)
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            t_loss += loss.item()
        print(f"    Epoch {ep}/{N_EPOCHS} | loss={t_loss/max(1,len(train_convs)):.4f}")

    modelo.eval()
    return modelo


def _get_or_train(train_ids, datos_por_sesion, tokenizer, cache) -> DialogXLClassifier:
    key = frozenset(train_ids)
    if key in cache:
        print(f"  ??  Modelo reutilizado | train={sorted(train_ids)}")
        return cache[key]

    mp = _model_path(train_ids)
    if os.path.exists(mp):
        modelo = DialogXLClassifier().to(DEVICE)
        modelo.load_state_dict(torch.load(mp, map_location=DEVICE,
                                          weights_only=True))
        modelo.eval()
        cache[key] = modelo
        print(f"  ?? Modelo desde disco | train={sorted(train_ids)}")
        return modelo

    print(f"  ?? Entrenando DialogXL | train={sorted(train_ids)} ...")
    train_convs = []
    for sid in train_ids:
        train_convs.extend(datos_por_sesion[sid])
    modelo = _entrenar_dialogxl(train_convs, tokenizer)
    torch.save(modelo.state_dict(), mp)
    cache[key] = modelo
    print(f"  ? Guardado ? {mp}")
    return modelo


def _inferir_logits_texto(modelo, convs, tokenizer) -> dict:
    """Fusiona excitement?happiness y devuelve logits/probs en 4 clases."""
    results = {}
    modelo.eval()
    with torch.no_grad():
        for conv in convs:
            ids, mask, spans = _tokenizar_dialogo(
                [u['texto'] for u in conv['utterances']], tokenizer)
            logits_raw, v_idx, _ = modelo(ids, mask, spans)
            if logits_raw is None: continue
            probs_raw = F.softmax(logits_raw, dim=-1).cpu().numpy()

            for i, vi in enumerate(v_idx):
                u = conv['utterances'][vi]
                if u['emocion_raw'] == 'fear': continue  # excluir fear

                p = probs_raw[i]
                # excitement ? happiness
                p_hap = p[EMO6_TO_IDX['happiness']] + p[EMO6_TO_IDX['excitement']]
                p4 = np.array([
                    p[EMO6_TO_IDX['anger']], p_hap,
                    p[EMO6_TO_IDX['sadness']], p[EMO6_TO_IDX['neutral']],
                ], dtype=np.float32)
                p4 /= (p4.sum() + 1e-8)
                logits4 = np.log(p4 + 1e-8).astype(np.float32)

                emo_real = ('happiness' if u['emocion_raw'] in ('exc', 'excitement')
                            else u['emocion_raw'])
                if emo_real not in EMOCIONES_VALIDAS: continue

                results[u['uid']] = {
                    'logits':           logits4,
                    'probs':            p4,
                    'emocion':          emo_real,
                    'emocion_original': u['emocion_raw'],
                    'session':          conv['session'],
                }
    return results


# ================================================================
# ?? NESTED LOSO
# ================================================================

def nested_loso_texto() -> dict:
    print("\n?? Cargando tokenizer XLNet...")
    tokenizer = XLNetTokenizer.from_pretrained(MODELO_BASE)

    print("?? Leyendo datos de sesiones...")
    datos_por_sesion = {}
    for sp in SESIONES:
        sn  = os.path.basename(sp)
        idx = int(sn.replace("Session", ""))
        datos_por_sesion[idx] = _leer_datos_sesion(sp)
        print(f"  {sn}: {len(datos_por_sesion[idx])} diálogos")

    model_cache = {}
    nested_out  = {}

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
                print(f"  ? K={K} J={J} ? {len(nested_out[K][J])} utts (ckpt)")
                continue

            train_ids = ([s for s in SESSION_IDS if s != K]       if J == K
                         else [s for s in SESSION_IDS if s not in (K, J)])

            modelo  = _get_or_train(train_ids, datos_por_sesion,
                                    tokenizer, model_cache)
            results = _inferir_logits_texto(modelo, datos_por_sesion[J], tokenizer)
            nested_out[K][J] = results

            with open(ckpt, 'wb') as f: pickle.dump(results, f)
            print(f"  ?? K={K} J={J} ? {len(results)} logits")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ?? Guardar + métricas ??????????????????????????????????????
    out_path = os.path.join(RUTA_NESTED, "nested_text_logits.pkl")
    with open(out_path, 'wb') as f: pickle.dump(nested_out, f)
    print(f"\n? Guardado: {out_path}")

    print(f"\n{'='*55}")
    print("?? Accuracy por fold (DialogXL ? diagnóstico LOSO)")
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
            titulo="Texto DialogXL ? GLOBAL",
        )

    return nested_out


if __name__ == "__main__":
    nested_text = nested_loso_texto()
    print("\n? 04_text_nested_loso completado")


# ================================================================
# ================================================================
# 05_fusion.py ? FUSIÓN TARDÍA (EmbraceNetFuzzy) NESTED LOSO
#   PREREQUISITO: 03_audio + 04_text + (01 ó 02) completados
#   SALIDA: fusion_best_predictions.csv + fusion_grid_results.csv
# ================================================================

import itertools
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)

# ?? Elegir fuente de video ???????????????????????????????????????
# 'vgg19' o 'hse' ? cambia según qué script de video ejecutaste
VIDEO_SOURCE = 'hse'   # ? modificar aquí

_VIDEO_PKL_DIR = RUTA_VIDEO_HSE if VIDEO_SOURCE == 'hse' else RUTA_VIDEO_VGG
_VIDEO_SUFFIX  = 'hse'           if VIDEO_SOURCE == 'hse' else 'vgg19'

AUDIO_PKL = os.path.join(RUTA_NESTED, "nested_audio_logits.pkl")
TEXT_PKL  = os.path.join(RUTA_NESTED, "nested_text_logits.pkl")

TIPOS = ['logits', 'probs']   # combinaciones del grid search


# ================================================================
# ?? UTILIDADES
# ================================================================

def _cargar_video(session_num: int) -> dict:
    sn = f"Session{session_num}"
    p  = os.path.join(_VIDEO_PKL_DIR, f"{sn}_embeddings_{_VIDEO_SUFFIX}.pkl")
    if not os.path.exists(p):
        print(f"  ??  Video pkl no encontrado: {p}"); return {}
    with open(p, 'rb') as f: return pickle.load(f)


def _get_vector(entry, tipo: str) -> np.ndarray:
    if entry is None: return np.zeros(NUM_CLASSES, dtype=np.float32)
    logits = entry.get('logits')
    probs  = entry.get('probs')
    if tipo == 'logits':
        v = logits if logits is not None else probs
    else:
        v = (probs if probs is not None
             else (F.softmax(torch.tensor(logits), 0).numpy()
                   if logits is not None else None))
    return v if v is not None else np.zeros(NUM_CLASSES, dtype=np.float32)
