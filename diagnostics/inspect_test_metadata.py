"""Inspect the metadata of test trials.
Who is in the first half vs second half?
"""
import sys
from pathlib import Path
track_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(track_dir))

from benchmark_utils.nb_task import load_task

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

test_info = []
for X_b, y_b, info in loaders["test"]:
    test_info.append(info)

print(f"Total test trials: {len(test_info)}")
print(f"Sample info[0]: {test_info[0]}")
print(f"Sample info[377]: {test_info[377]}")
print(f"Sample info[-1]: {test_info[-1]}")

# Collect unique values of keys in info
keys = test_info[0].keys() if hasattr(test_info[0], 'keys') else []
print(f"Info keys: {keys}")

if isinstance(test_info[0], dict):
    for k in keys:
        vals = [str(inf.get(k)) for inf in test_info]
        unique_vals = list(dict.fromkeys(vals))
        print(f"Key '{k}': {len(unique_vals)} unique values -> {unique_vals[:5]}")

