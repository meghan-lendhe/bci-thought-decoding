"""Package Adaptive Guarded Hierarchical Solver into Codabench Submission Archive.

Produces:
- submission.py (self-contained, robust)
- weights.joblib (pre-fitted weights on 1,521 training trials)
- submission_adaptive_guarded_hierarchical.zip
"""

import sys
import shutil
import zipfile
from pathlib import Path
import joblib
import numpy as np

track_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task
from solvers.adaptive_guarded_hierarchical import (
    AdaptiveGuardedHierarchicalModel,
    Solver
)

output_bundle_dir = track_dir / "submission_bundle_temp"
if output_bundle_dir.exists():
    shutil.rmtree(output_bundle_dir)
output_bundle_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PACKAGING ADAPTIVE GUARDED HIERARCHICAL SUBMISSION")
print("=" * 80)

# 1. Load train data
loaders, meta = load_task(
    "eeg",
    "_motor_imagery_stream",
    data_dir=track_dir / "data" / "neural_compet",
    device="cpu",
    batch_size=1,
    seed=42,
    num_workers=0,
    subset="full",
    target_transform=lambda y: y.argmax(-1) if hasattr(y, "argmax") else y,
)

# 2. Initialize model with optimized parameters
model = AdaptiveGuardedHierarchicalModel(
    n_chans=meta.get("n_chans", 41),
    n_classes=meta.get("n_classes", 3),
    sfreq=meta.get("sfreq", 120.0),
    prior_N0=25,
    base_th_raw=0.01,
    guard_raw=0.60,
    w_lat_raw=2.0,
    M_mot=10,
    M_cog=10,
    M_lat=15,
    device="cpu",
)

# 3. Fit on training loader
print("\n[1/4] Fitting model on 1,521 training trials...")
model.fit(loaders["train"])

# 4. Save weights.joblib
solver = Solver()
print(f"\n[2/4] Saving weights.joblib to {output_bundle_dir}...")
solver.save_model(model, output_bundle_dir)

# 5. Copy self-contained submission.py
print("\n[3/4] Generating submission.py...")
solver_source = (track_dir / "solvers" / "adaptive_guarded_hierarchical.py").read_text()
(output_bundle_dir / "submission.py").write_text(solver_source)

# 6. Create Zip Archive
zip_dest = track_dir.parent / "submission_adaptive_guarded_hierarchical.zip"
print(f"\n[4/4] Creating submission archive: {zip_dest}...")
with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(output_bundle_dir / "submission.py", arcname="submission.py")
    zf.write(output_bundle_dir / "weights.joblib", arcname="weights.joblib")

print(f"Archive created cleanly! File size: {zip_dest.stat().st_size / 1024 / 1024:.2f} MB")
