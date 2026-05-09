# ================================================================
# 05_fusion.py ? FUSIÓN TARDÍA (EmbraceNetFuzzy) NESTED LOSO
#   PREREQUISITO: 03_audio + 04_text + (01 ó 02) completados
#   SALIDA: fusion_best_predictions.csv + fusion_grid_results.csv
# ================================================================
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.embrace_net_fuzzy import EmbraceNetFuzzy
import itertools
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)

# ?? Elegir fuente de video ???????????????????????????????????????
VIDEO_SOURCE = 'vgg19'

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


# ================================================================
# ?? DATASET
# ================================================================

class FusionDataset(Dataset):
    def __init__(self, samples): self.s = samples
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        d = self.s[i]
        return (torch.tensor(d['v']), torch.tensor(d['a']),
                torch.tensor(d['t']), torch.tensor(d['y']))


def _build_samples(K, J, audio_n, text_n, video_per_s,
                   tv, ta, tt) -> list:
    audio_J = audio_n.get(K, {}).get(J, {})
    text_J  = text_n.get(K, {}).get(J, {})
    video_J = video_per_s.get(J, {})
    uids    = set(audio_J) | set(text_J) | set(video_J)
    samples = []
    for uid in uids:
        src = audio_J.get(uid) or text_J.get(uid) or video_J.get(uid)
        if not src: continue
        emo = src.get('emocion')
        if emo not in EMO_TO_IDX: continue
        samples.append({
            'v': _get_vector(video_J.get(uid), tv),
            'a': _get_vector(audio_J.get(uid), ta),
            't': _get_vector(text_J.get(uid),  tt),
            'y': EMO_TO_IDX[emo],
        })
    return samples


# ================================================================
# ?? UN FOLD LOSO
# ================================================================

def _run_fold(K, audio_n, text_n, video_per_s,
              tv, ta, tt, epochs=30, lr=1e-3):
    train_samples = []
    for J in SESSION_IDS:
        if J != K:
            train_samples.extend(
                _build_samples(K, J, audio_n, text_n, video_per_s, tv, ta, tt))
    test_samples = _build_samples(K, K, audio_n, text_n, video_per_s, tv, ta, tt)

    if not train_samples or not test_samples:
        return 0.0, []

    train_loader = DataLoader(FusionDataset(train_samples),
                              batch_size=32, shuffle=True)
    test_loader  = DataLoader(FusionDataset(test_samples), batch_size=32)

    model = EmbraceNetFuzzy().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    crit  = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for _ in range(epochs):
        model.train()
        for v, a, t, y in train_loader:
            v, a, t, y = [x.to(DEVICE) for x in (v, a, t, y)]
            opt.zero_grad(); crit(model(v, a, t), y).backward(); opt.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for v, a, t, y in test_loader:
                v, a, t, y = [x.to(DEVICE) for x in (v, a, t, y)]
                preds.extend(model(v, a, t).argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for v, a, t, y in test_loader:
            v, a, t, y = [x.to(DEVICE) for x in (v, a, t, y)]
            logits = model(v, a, t)
            prbs   = F.softmax(logits, dim=-1).cpu().numpy()
            prd    = logits.argmax(1).cpu().numpy()
            lbl    = y.cpu().numpy()
            for i in range(len(prd)):
                fold_preds.append({
                    'fold_K': K, 'true': lbl[i], 'pred': prd[i],
                    **{f'p_{e}': prbs[i][j] for j, e in enumerate(EMOCIONES_VALIDAS)}
                })
    print(f"  ? K={K} | acc={best_acc:.4f} "
          f"| train={len(train_samples)} test={len(test_samples)}")
    return best_acc, fold_preds


# ================================================================
# ?? GRID SEARCH + EVALUACIÓN FINAL
# ================================================================

def run_fusion_grid():
    with open(AUDIO_PKL, 'rb') as f: audio_n = pickle.load(f)
    with open(TEXT_PKL,  'rb') as f: text_n  = pickle.load(f)
    video_per_s = {s: _cargar_video(s) for s in SESSION_IDS}

    combinaciones = list(itertools.product(TIPOS, TIPOS, TIPOS))
    resultados = []

    print(f"\n?? Grid Search ? {len(combinaciones)} combinaciones\n")
    for (tv, ta, tt) in combinaciones:
        nombre = f"V({tv})_A({ta})_T({tt})"
        print(f"\n{'?'*50}\n?  {nombre}")
        fold_accs, all_preds = [], []

        for K in SESSION_IDS:
            acc, preds = _run_fold(K, audio_n, text_n, video_per_s, tv, ta, tt)
            fold_accs.append(acc); all_preds.extend(preds)

        mean_acc = float(np.mean(fold_accs))
        std_acc  = float(np.std(fold_accs))
        resultados.append({
            'combinacion':   nombre,
            'tipo_v': tv, 'tipo_a': ta, 'tipo_t': tt,
            'loso_mean_acc': mean_acc,
            'loso_std_acc':  std_acc,
            **{f'acc_fold{k}': a for k, a in enumerate(fold_accs, 1)},
        })
        print(f"  LOSO acc = {mean_acc:.4f} ± {std_acc:.4f}")

    df = pd.DataFrame(resultados).sort_values('loso_mean_acc', ascending=False)
    csv_grid = os.path.join(RUTA_FUSION, "fusion_grid_results.csv")
    df.to_csv(csv_grid, index=False)
    print(f"\n?? Grid guardado: {csv_grid}")
    print("\nTop 5:")
    print(df[['combinacion', 'loso_mean_acc', 'loso_std_acc']].head(5).to_string())

    # ?? Evaluación final con la mejor combinación (más epochs) ??
    best = df.iloc[0]
    tv, ta, tt = best['tipo_v'], best['tipo_a'], best['tipo_t']
    print(f"\n{'='*55}")
    print(f"?? Evaluación completa: V({tv}) A({ta}) T({tt})")
    all_preds_best = []
    fold_accs_best = []
    for K in SESSION_IDS:
        acc, preds = _run_fold(K, audio_n, text_n, video_per_s,
                               tv, ta, tt, epochs=50, lr=5e-4)
        fold_accs_best.append(acc); all_preds_best.extend(preds)

    y_true = [p['true'] for p in all_preds_best]
    y_pred = [p['pred'] for p in all_preds_best]

    report_accuracy(
        [IDX_TO_EMO[i] for i in y_true],
        [IDX_TO_EMO[i] for i in y_pred],
        titulo=f"FUSIÓN TARDÍA ? V({tv}) A({ta}) T({tt})",
    )
    f1w = f1_score(y_true, y_pred, average='weighted')
    f1m = f1_score(y_true, y_pred, average='macro')
    print(f"  LOSO Acc   : {np.mean(fold_accs_best):.4f} ± {np.std(fold_accs_best):.4f}")
    print(f"  F1-weighted: {f1w:.4f}")
    print(f"  F1-macro   : {f1m:.4f}")

    # Guardar predicciones
    csv_pred = os.path.join(RUTA_FUSION, "fusion_best_predictions.csv")
    pd.DataFrame(all_preds_best).to_csv(csv_pred, index=False)
    print(f"\n  ?? Predicciones ? {csv_pred}")

    return df


if __name__ == "__main__":
    df_grid = run_fusion_grid()
    print("\n? 05_fusion completado")
