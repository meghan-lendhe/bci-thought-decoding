"""Diagnostic: Error Anatomy & Train vs Test Separation.

Answers:
1. Is this Underfitting or Cross-Subject Domain Shift?
   - What is the Training Confusion Matrix (1,521 trials)?
   - What is the Test Confusion Matrix (755 trials)?
2. The 2-Stage Decomposition:
   - Stage 1: Motor (Class 0) vs Cognitive (Classes 1 & 2) accuracy.
   - Stage 2: Math (Class 1) vs Word (Class 2) accuracy (given it's Cognitive).
3. Test Stream Subject & Temporal Profile:
   - Who is in the test stream? (Recordings, subjects, sessions).
   - Does Word Gen accuracy decay over the test session (early vs late)?
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task
from solvers.adaptive_regional_ensemble import AdaptiveRegionalEnsembleModel

print("=" * 75)
print("DIAGNOSTIC: ERROR ANATOMY & TRAIN VS TEST SEPARATION")
print("=" * 75)

data_dir = track_dir / "data" / "neural_compet"
loaders, meta = load_task(
    "eeg",
    "_motor_imagery_stream",
    data_dir=data_dir,
    dataset=None,
    device="cpu",
    batch_size=1,
    seed=42,
    num_workers=0,
    subset="full",
    target_transform=lambda y: y.argmax(-1) if hasattr(y, "argmax") else y,
)

model = AdaptiveRegionalEnsembleModel(
    n_chans=meta.get("n_chans", 41),
    n_classes=meta.get("n_classes", 3),
    sfreq=meta.get("sfreq", 120.0),
    alpha=750.0,
    prior_N0=15,
    device="cpu",
)

# 1. Fit on Training Data
print("\n[1/4] Fitting Regional Subsystems Pipeline on 1,521 training trials...")
model.fit(loaders["train"])

# 2. Evaluate Training Set Performance (Underfitting vs Overfitting Check)
print("\n[2/4] Evaluating Performance on TRAINING Set (Checking Representation Capacity)...")
train_preds = model.ridge.predict(model.X_train_scaled)
train_cm = confusion_matrix(model.y_train, train_preds, labels=[0, 1, 2])
train_acc = (train_preds == model.y_train).mean() * 100.0

print(f"Overall Training Accuracy: {train_acc:.2f}% ({ (train_preds == model.y_train).sum() }/{len(model.y_train)})")
print(f"• Motor Imagery (0) Train Acc:    {train_cm[0, 0]/train_cm[0].sum()*100:.1f}% ({train_cm[0, 0]}/{train_cm[0].sum()})")
print(f"• Mental Subtraction (1) Train Acc: {train_cm[1, 1]/train_cm[1].sum()*100:.1f}% ({train_cm[1, 1]}/{train_cm[1].sum()})")
print(f"• Word Generation (2) Train Acc:  {train_cm[2, 2]/train_cm[2].sum()*100:.1f}% ({train_cm[2, 2]}/{train_cm[2].sum()})")
print(f"Training Confusion Matrix:\n{train_cm}")

# 3. Stream Test Set & Capture Detailed Trial-by-Trial Metadata
print("\n[3/4] Streaming Test Set with Metadata Tracking...")
model.reset_state()

y_test_list = []
y_pred_list = []
info_list = []
dec_scores_list = []

for idx, (X_b, y_b, info) in enumerate(loaders["test"]):
    y_test_list.append(int(y_b.cpu().numpy()[0]))
    info_list.append(info)
    
    # Predict and capture continuous decision scores
    X_trial = X_b.cpu().numpy()
    # We want both ridge and transformer scores
    split = X_trial.shape[-1] // 2
    w_e = X_trial[np.newaxis, ..., :split]
    w_l = X_trial[np.newaxis, ..., split:]
    
    # Use model's predict
    p = model.predict(X_b)
    y_pred_list.append(p[0])

y_true = np.array(y_test_list)
y_pred = np.array(y_pred_list)
classes = [0, 1, 2]
class_names = ["Motor Imagery (0)", "Mental Sub (1)   ", "Word Gen (2)     "]

cm = confusion_matrix(y_true, y_pred, labels=classes)

print("\n" + "=" * 75)
print("TEST CONFUSION MATRIX:")
print("=" * 75)
print(f"{'Class':<22} | {'Correct':<10} | {'Total':<10} | {'Accuracy':<10}")
print("-" * 75)
for c, cname in zip(classes, class_names):
    cor = cm[c, c]
    tot = cm[c].sum()
    print(f"{cname:<22} | {cor:<10} | {tot:<10} | {cor/tot*100:6.1f}%")
print("-" * 75)
print(f"Confusion Matrix (True \\ Pred):\n{cm}")

# 4. Two-Stage Decision Analysis
print("\n" + "=" * 75)
print("STAGE 1: MOTOR (Class 0) VS COGNITIVE (Classes 1 & 2)")
print("=" * 75)
is_true_motor = (y_true == 0)
is_pred_motor = (y_pred == 0)

motor_tp = (is_true_motor & is_pred_motor).sum()
motor_fn = (is_true_motor & ~is_pred_motor).sum()
cog_tp = (~is_true_motor & ~is_pred_motor).sum()
cog_fn = (~is_true_motor & is_pred_motor).sum()

motor_recall = motor_tp / (motor_tp + motor_fn) * 100.0
cog_recall = cog_tp / (cog_tp + cog_fn) * 100.0
binary_acc = (motor_tp + cog_tp) / len(y_true) * 100.0

print(f"• Motor Sensitivity (MI correct):             {motor_recall:.1f}% ({motor_tp}/{motor_tp+motor_fn})")
print(f"• Cognitive Sensitivity (Cognitive not MI):   {cog_recall:.1f}% ({cog_tp}/{cog_tp+cog_fn})")
print(f"• Overall Binary Motor vs Cognitive Accuracy: {binary_acc:.1f}% ({motor_tp+cog_tp}/{len(y_true)})")
print(f"• Cognitive Leakage into Motor:               {cog_fn}/{cog_tp+cog_fn} cognitive trials mistaken for hand movement!")

print("\n" + "=" * 75)
print("STAGE 2: MATH (Class 1) VS WORD (Class 2) AMONG COGNITIVE TRIALS")
print("=" * 75)
cog_mask = (y_true != 0) & (y_pred != 0)  # Trials that are cognitive AND predicted as cognitive
y_true_cog = y_true[cog_mask]
y_pred_cog = y_pred[cog_mask]

math_in_cog = (y_true_cog == 1)
word_in_cog = (y_true_cog == 2)

math_correct_in_cog = (y_pred_cog[math_in_cog] == 1).sum()
math_total_in_cog = math_in_cog.sum()
word_correct_in_cog = (y_pred_cog[word_in_cog] == 2).sum()
word_total_in_cog = word_in_cog.sum()

math_acc_given_cog = math_correct_in_cog / math_total_in_cog * 100.0 if math_total_in_cog > 0 else 0.0
word_acc_given_cog = word_correct_in_cog / word_total_in_cog * 100.0 if word_total_in_cog > 0 else 0.0
cog_2class_acc = (math_correct_in_cog + word_correct_in_cog) / len(y_true_cog) * 100.0 if len(y_true_cog) > 0 else 0.0

print(f"Among the {len(y_true_cog)} trials the model recognized as cognitive:")
print(f"• Math vs Word 2-class Accuracy:  {cog_2class_acc:.1f}%")
print(f"• Math Accuracy (given cognitive): {math_acc_given_cog:.1f}% ({math_correct_in_cog}/{math_total_in_cog})")
print(f"• Word Accuracy (given cognitive): {word_acc_given_cog:.1f}% ({word_correct_in_cog}/{word_total_in_cog})")

# 5. Temporal Drift: First Half vs Second Half of Test Session
print("\n" + "=" * 75)
print("STAGE 3: TEMPORAL DRIFT ACROSS THE TEST STREAM")
print("=" * 75)
n_half = len(y_true) // 2

acc_h1 = (y_pred[:n_half] == y_true[:n_half]).mean() * 100.0
acc_h2 = (y_pred[n_half:] == y_true[n_half:]).mean() * 100.0

w_mask = (y_true == 2)
w_h1 = (y_pred[:n_half][w_mask[:n_half]] == 2).mean() * 100.0
w_h2 = (y_pred[n_half:][w_mask[n_half:]] == 2).mean() * 100.0

m_mask = (y_true == 1)
m_h1 = (y_pred[:n_half][m_mask[:n_half]] == 1).mean() * 100.0
m_h2 = (y_pred[n_half:][m_mask[n_half:]] == 1).mean() * 100.0

print(f"• First 377 Trials  (Minutes 0-12): Total Acc = {acc_h1:.1f}% | Math Acc = {m_h1:.1f}% | Word Acc = {w_h1:.1f}%")
print(f"• Second 378 Trials (Minutes 12-25): Total Acc = {acc_h2:.1f}% | Math Acc = {m_h2:.1f}% | Word Acc = {w_h2:.1f}%")
print("=" * 75)
