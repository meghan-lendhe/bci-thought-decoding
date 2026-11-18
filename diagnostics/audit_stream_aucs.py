import numpy as np
from sklearn.metrics import roc_auc_score

data = np.load("scratch/stream_full_traces.npz")
streams = data["streams"]
y_true = data["y_true"]
ms_raw = data["ms_raw"]
raw01 = data["raw01"]
raw02 = data["raw02"]
raw12 = data["raw12"]
lat_val = data["lat_val"]

print("=" * 80)
print("SPECIALIST DISCRIMINATIVE POWER (ROC-AUC) ACROSS STREAMS")
print("=" * 80)

for sid in [3, 4, 5, 10]:
    m = streams == sid
    yt = y_true[m]
    
    # Motor vs Rest (Class 0 vs Class 1, 2)
    y_mot = (yt == 0).astype(int)
    auc_mot = roc_auc_score(y_mot, ms_raw[m])
    
    # 0v1 Specialist on {0, 1}
    m01 = m & np.isin(y_true, [0, 1])
    # raw01 < 0 -> 0, > 0 -> 1. Invert so 0 has higher score
    auc_01 = roc_auc_score((y_true[m01] == 0).astype(int), -raw01[m01])
    
    # 0v2 Specialist on {0, 2}
    m02 = m & np.isin(y_true, [0, 2])
    auc_02 = roc_auc_score((y_true[m02] == 0).astype(int), -raw02[m02])
    
    # 1v2 Specialist on {1, 2}
    m12 = m & np.isin(y_true, [1, 2])
    # raw12 < 0 -> 1, > 0 -> 2. Class 2 higher
    auc_12 = roc_auc_score((y_true[m12] == 2).astype(int), raw12[m12])
    
    # Lateralization on {1, 2} (Class 1 Math has higher lat, so Class 2 has lower)
    auc_lat = roc_auc_score((y_true[m12] == 1).astype(int), lat_val[m12])
    
    print(f"\n--- STREAM {sid} ---")
    print(f"  Motor vs Rest (ms_raw):  AUC = {auc_mot:.3f}")
    print(f"  Motor vs Math (raw01):   AUC = {auc_01:.3f}")
    print(f"  Motor vs Word (raw02):   AUC = {auc_02:.3f}")
    print(f"  Math vs Word  (raw12):   AUC = {auc_12:.3f}")
    print(f"  Math vs Word  (lat):     AUC = {auc_lat:.3f}")
