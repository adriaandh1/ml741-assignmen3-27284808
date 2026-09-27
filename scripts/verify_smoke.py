"""Run repeated smoke tests and verify bundles, predictions, and determinism.

Usage: python scripts/verify_smoke.py --output tmp/readiness-review
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ml741_assignment3.artifacts import file_sha256, load_model_checkpoint, write_json
from ml741_assignment3.data import make_smoke_dataset, stratified_splits
from ml741_assignment3.smoke import run_smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = {}
    outcomes = []
    for name, seed in [
        ("seed741", 741),
        ("seed742", 742),
        ("seed743", 743),
        ("repeat741", 741),
    ]:
        destination = args.output / name
        results = run_smoke(destination, seed=seed, config_path="configs/smoke.yaml")
        x, y = make_smoke_dataset(seed=seed)
        splits = stratified_splits(x, y, seed=seed)
        for result in results:
            bundle = destination / "runs" / result.method
            checksums = json.loads((bundle / "checksums.json").read_text())
            assert all(file_sha256(bundle / p) == digest for p, digest in checksums.items())
            model, order = load_model_checkpoint(bundle / "checkpoints/final.pt")
            with torch.no_grad():
                positions = model(torch.from_numpy(splits.x_test)).argmax(1).numpy()
            np.testing.assert_array_equal(np.asarray(order)[positions], result.predictions)
            assert np.isfinite([r["validation_loss"] for r in result.epoch_records]).all()
            assert len(result.class_order) == 4
            if name == "seed741":
                original[result.method] = result
            elif name == "repeat741":
                previous = original[result.method]
                assert previous.epoch_records == result.epoch_records
                assert previous.events == result.events
                np.testing.assert_array_equal(previous.predictions, result.predictions)
                for key in previous.model_state:
                    assert torch.equal(previous.model_state[key], result.model_state[key])
            growth = sum(e["event"] == "hidden_unit_added" for e in result.events)
            if result.method == "adaptive_capacity_incremental":
                assert growth > 0
            outcomes.append(
                {
                    "run": name,
                    "method": result.method,
                    "macro_f1": result.metrics["macro_f1"],
                    "growth_events": growth,
                    "checkpoint_predictions_match": True,
                    "checksums_valid": True,
                }
            )
        print(f"Verified {name}: all three methods, bundles and checkpoint predictions")
    write_json(
        args.output / "verification.json",
        {
            "runs": outcomes,
            "seed741_bitwise_repeatability": True,
        },
    )


if __name__ == "__main__":
    main()
