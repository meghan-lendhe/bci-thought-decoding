"""Reproduce Codabench Submission Execution Locally.

Simulates the exact Codabench runner:
1. Extracts submission_adaptive_master_ensemble.zip into an isolated folder.
2. Imports submission.py and calls solver.load_model(meta).
3. Executes model.reset_state() and model.predict(X).
"""

import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import torch

if len(sys.argv) > 1:
    zip_path = Path(sys.argv[1]).resolve()
else:
    zip_path = Path("submission_adaptive_regional_ensemble.zip").resolve()
    if not zip_path.exists():
        zip_path = Path("2026-competition/submission_adaptive_regional_ensemble.zip").resolve()

print("=" * 65)
print("TESTING SUBMISSION ARCHIVE LOCALLY...")
print(f"Archive: {zip_path}")
print("=" * 65)

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    print(f"\n[1/3] Extracting archive to isolated sandbox: {tmp_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_path)
    
    # Prepend isolated sandbox to sys.path
    sys.path.insert(0, str(tmp_path))
    track_dir = Path(__file__).resolve().parent
    sys.path.insert(1, str(track_dir))
    
    print("\n[2/3] Importing Solver and calling load_model()...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("sub", tmp_path / "submission.py")
    sub_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub_module)
    
    solver = sub_module.Solver()
    meta = {
        "submission_dir": tmp_path,
        "n_chans": 41,
        "n_classes": 3,
        "sfreq": 120.0,
        "device": "cpu",
    }
    
    try:
        model = solver.load_model(meta)
        print("      [SUCCESS] Model loaded cleanly!")
    except Exception as e:
        print(f"\n[CRITICAL FAILURE IN load_model]: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\n[3/3] Testing reset_state() and predict(X)...")
    try:
        if hasattr(model, "reset_state"):
            model.reset_state()
            print("      [OK] reset_state() executed!")
        
        # Test window shape (1, 41, 480)
        X_dummy = torch.randn(1, 41, 480)
        pred = model.predict(X_dummy)
        print(f"      [SUCCESS] predict(X) returned class: {pred} (type: {type(pred)}, shape: {pred.shape})")
    except Exception as e:
        print(f"\n[CRITICAL FAILURE IN predict]: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 65)
print("ALL CONTRACT CHECKS PASSED PERFECTLY!")
print("=" * 65)

