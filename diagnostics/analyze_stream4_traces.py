import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

data = np.load("scratch/stream_full_traces.npz")
streams = data["streams"]
y_true = data["y_true"]
y_pred = data["y_pred"]
ms_raw = data["ms_raw"]
th_mot = data["th_mot"]
th_cog = data["th_cog"]
eff_12 = data["eff_12"]
raw01 = data["raw01"]
raw02 = data["raw02"]
raw12 = data["raw12"]
lat_val = data["lat_val"]
z_lat = data["z_lat"]
gate_open = data["gate_open"]
guard_triggered = data["guard_triggered"]

print("=" * 80)
print("DEEP DIAGNOSTIC OF STREAM 4 (Subject 4, 42.80% BalAcc) VS STREAM 10 (65.68%)")
print("=" * 80)

for sid in [4, 10]:
    mask = streams == sid
    yt = y_true[mask]
    yp = y_pred[mask]
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
    acc = balanced_accuracy_score(yt, yp) * 100.0
    print(f"\n{'='*30} STREAM {sid} ({acc:.2f}% BalAcc, N={len(yt)}) {'='*30}")
    print("Confusion Matrix (True rows, Pred cols):")
    print(cm)
    
    for c in [0, 1, 2]:
        cname = ["Motor (0)", "Math (1)", "Word (2)"][c]
        mc = mask & (y_true == c)
        n_c = mc.sum()
        
        # Breakdown of decisions
        n_pred0 = (y_pred[mc] == 0).sum()
        n_pred1 = (y_pred[mc] == 1).sum()
        n_pred2 = (y_pred[mc] == 2).sum()
        
        # Gate & Guard stats
        n_gate = gate_open[mc].sum()
        n_veto = guard_triggered[mc].sum()
        
        mean_ms = ms_raw[mc].mean()
        mean_th_mot = th_mot[mc].mean()
        mean_raw12 = raw12[mc].mean()
        mean_th_cog = th_cog[mc].mean()
        mean_eff12 = eff_12[mc].mean()
        mean_lat = lat_val[mc].mean()
        mean_zlat = z_lat[mc].mean()
        
        print(f"\n--- {cname} (N={n_c}) ---")
        print(f"  Predictions: Motor={n_pred0} ({n_pred0/n_c*100:.1f}%), Math={n_pred1} ({n_pred1/n_c*100:.1f}%), Word={n_pred2} ({n_pred2/n_c*100:.1f}%)")
        print(f"  Gate Open (ms > th_mot): {n_gate}/{n_c} ({n_gate/n_c*100:.1f}%) | Guard Veto: {n_veto}/{n_c} ({n_veto/n_c*100:.1f}%)")
        print(f"  Motor Score:       Mean ms={mean_ms:+.4f} vs Mean th_mot={mean_th_mot:+.4f} (Margin: {mean_ms - mean_th_mot:+.4f})")
        print(f"  Cognitive raw12:   Mean raw12={mean_raw12:+.4f} vs Mean th_cog={mean_th_cog:+.4f}")
        print(f"  Effective eff_12:  Mean eff_12={mean_eff12:+.4f} (|eff_12| mean = {np.abs(eff_12[mc]).mean():.4f})")
        print(f"  Lateralization:    Mean lat={mean_lat:+.4f}, Mean z_lat={mean_zlat:+.4f}")
