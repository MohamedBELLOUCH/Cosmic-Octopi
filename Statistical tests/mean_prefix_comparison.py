"""Apply the existing three mean tests to prefixes of saved trajectories.

No simulation or training. Statistical implementations and settings match the
cutoff comparison; only the selected time windows differ.
"""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from mean_comparison import (TEST_NAMES, group_summary, holm, mats_test,
                             pillai_test, save, welch_james_test)


HERE = Path(__file__).resolve().parent


def analysis_signature(config, source_hash):
    code = Path(__file__).read_bytes() + (HERE / "mean_comparison.py").read_bytes()
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()
                          + source_hash.encode() + code).hexdigest()


def main():
    config = json.loads((HERE / "setup_mean_prefix_comparison.json").read_text(encoding="utf-8"))
    source = HERE / config["input_result_file"]
    target = HERE / config["output_file"]
    if source.resolve() == target.resolve():
        raise ValueError("Analysis output must differ from trajectory input")
    raw_bytes = source.read_bytes()
    data = pickle.loads(raw_bytes)
    if not data["completed_runs"].all() or data["setup"].get("simulated_rewards") is not True:
        raise ValueError("Complete simulated trajectories are required")
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    signature = analysis_signature(config, source_hash)
    x = data["costs"] / data["objective_scale"]
    horizons = np.array(config["horizons"], dtype=int)
    if not len(horizons) or np.any(horizons < 2) or np.any(horizons > x.shape[1]):
        raise ValueError("Invalid prefix horizons")
    if not np.array_equal(horizons, np.unique(horizons)):
        raise ValueError("Prefix horizons must be increasing and distinct")
    n_arms = len(data["arm_values"])
    if target.exists():
        result = pickle.loads(target.read_bytes())
        if result["signature"] != signature:
            raise ValueError("Settings, code or input changed; choose a new output file")
        if result["complete"]:
            print(f"Already complete: {target}")
            return
    else:
        shape = (3, len(horizons), n_arms)
        result = {"configuration": config, "signature": signature, "source_sha256": source_hash,
                  "complete": False, "test_names": TEST_NAMES, "horizons": horizons,
                  "window_lengths": horizons.copy(), "final_iteration": x.shape[1],
                  "n_trajectories": len(x), "arm_values": data["arm_values"],
                  "statistics": np.full(shape, np.nan), "pvalues": np.full(shape, np.nan),
                  "response_dimensions": np.zeros(shape[1:], int),
                  "welch_james_rank": np.zeros(shape[1:], int),
                  "welch_james_full_rank": np.zeros(shape[1:], int)}
    with threadpool_limits(limits=2), tqdm(total=len(horizons)*n_arms,
            initial=int(np.isfinite(result["pvalues"]).all(axis=0).sum()),
            desc="Mean tests on standard horizons", ascii=True, mininterval=2) as progress:
        for hi, horizon in enumerate(horizons):
            window = x[:, :horizon]
            for arm in range(n_arms):
                if np.isfinite(result["pvalues"][:, hi, arm]).all():
                    continue
                mask = data["arm_indices"][:, :horizon] == arm
                means, cov, counts, varying = group_summary(window, mask)
                pillai = pillai_test(means, cov, counts)
                welch = welch_james_test(means, cov, counts, config["covariance_rank_tolerance"])
                mats = mats_test(means, cov, counts, config["mats_bootstrap_draws"],
                    np.random.SeedSequence([config["seed"], 1, int(horizon), arm]), config["bootstrap_batch_size"])
                for i, test in enumerate([pillai, welch, mats]):
                    result["statistics"][i, hi, arm], result["pvalues"][i, hi, arm] = test[:2]
                result["response_dimensions"][hi, arm] = varying.sum()
                result["welch_james_rank"][hi, arm], result["welch_james_full_rank"][hi, arm] = welch[2:]
                save(target, result)
                progress.update()
    result["adjusted_pvalues"] = np.array([[holm(p) for p in test] for test in result["pvalues"]])
    result["rejection_percentage"] = 100*(result["adjusted_pvalues"] <= config["alpha"]).mean(axis=-1)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Saved trajectory input changed during analysis")
    result["complete"] = True
    save(target, result)
    print("Horizon | MANOVA | Welch-James | MATS")
    for hi, horizon in enumerate(horizons):
        print(f"{horizon:7d} | " + " | ".join(f"{v:.1f}%" for v in result["rejection_percentage"][:, hi]))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
