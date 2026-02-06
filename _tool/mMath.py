import numpy as np
from sklearn.metrics import f1_score
dis_eucid = lambda x, y: np.linalg.norm(x - y)
def f1_multicls(preds, truths, n_classes):
    preds = np.vstack(preds)
    truths = np.concatenate(truths)
    preds_label = np.argmax(preds, axis=-1)
    micro_f1 = f1_score(truths, preds_label, average='micro', labels=np.arange(n_classes).tolist(), zero_division=np.nan)
    macro_f1 = f1_score(truths, preds_label, average='macro', labels=np.arange(n_classes).tolist(), zero_division=np.nan)
    return {'Mi-F1': round(micro_f1, 5), 'Ma-F1': round(macro_f1, 5)}
