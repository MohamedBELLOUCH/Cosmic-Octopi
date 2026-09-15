"""Empirical Xi calibration after int8 uploads; execution explicitly starts PPO.

Uses the project's PPO components, quantized upload codec, FedAvg and Xi ridge
regression. No source Xi is used to generate observations. --check is read-only
and creates no environments. --resume restarts at the last complete wet experiment.
"""

import argparse
from contextlib import redirect_stdout
import copy
import importlib.util
import io
from itertools import combinations
import json
import math
from pathlib import Path
import pickle
import random
import sys

HERE = Path(__file__).resolve().parent
CLASS_NAMES = ("Cheetah", "Ant", "Leg", "Humanoid")


def validate_setup(s):
    if s["simulated_rewards"] is not False:
        raise ValueError("Quantization calibration requires measured PPO rewards")
    if not s["class_names"] or len(set(s["class_names"])) != len(s["class_names"]):
        raise ValueError("class_names must be nonempty and unique")
    if any(c not in CLASS_NAMES for c in s["class_names"]):
        raise ValueError("Unknown environment class")
    for key in ("n_agents", "n_wet_experiments", "n_episodes"):
        if type(s[key]) is not int or s[key] < 2:
            raise ValueError(f"{key} must be an integer >= 2")
    if s["n_agents"] > 12:
        raise ValueError("All-subset calibration is limited to 12 agents")
    if s["gravity_mean"] <= 0 or s["gravity_variance"] <= 0:
        raise ValueError("Gravity mean and variance must be positive")
    lo, hi = s["warmup_episode_bounds"]
    if type(lo) is not int or type(hi) is not int or not 1 <= lo <= hi:
        raise ValueError("Invalid warmup episode bounds")
    fixed = {"subset_design": "all_nonempty", "target_agent_selection": "uniform_from_subset",
             "aggregation": "episode_weighted_federated_average",
             "fitness_definition": "arithmetic_mean_of_contributors_mean_warmup_returns_at_zero_freshness"}
    for key, value in fixed.items():
        if s[key] != value:
            raise ValueError(f"Unsupported {key}: {s[key]}")
    q = s["quantization"]
    if q["bits"] != 8 or q["scheme"] != "symmetric_per_tensor":
        raise ValueError("The transport codec implements symmetric per-tensor int8")
    if s["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if type(s["seed"]) is not int or s["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for class_name in s["class_names"]:
        env = s["environments"][class_name]
        if not env["id"].endswith("-v5") or type(env["hidden_size"]) is not int or env["hidden_size"] < 1:
            raise ValueError(f"Invalid v5 environment/network for {class_name}")
        if not isinstance(env["kwargs"], dict):
            raise ValueError("Environment kwargs must be a dictionary")
    for key in ("updates_per_episode", "evaluation_episodes", "rollout_steps",
                "sampler_epochs", "minibatch_size"):
        if type(s["training"][key]) is not int or s["training"][key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if s["training"]["evaluation_action_mode"] != "sampled":
        raise ValueError("Evaluation follows the existing sampled-action policy")
    for key in ("learning_rate", "adam_eps", "cliprange", "value_loss_coefficient", "max_grad_norm"):
        if not math.isfinite(s["training"][key]) or s["training"][key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if not 0 < s["training"]["gamma"] <= 1 or not 0 <= s["training"]["gae_lambda"] <= 1:
        raise ValueError("Invalid GAE discount/trace parameters")
    # Preserve the reward scale used by Octopus rather than silently shifting it.
    if s["training"]["reward_offset"] != 0:
        raise ValueError("reward_offset must remain zero to match the quantized Octopus")
    for bounds in s["ou_fit"]["bounds"].values():
        if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
            raise ValueError("OU bounds must be positive and ordered")
    if s["ou_fit"]["dt"] <= 0 or s["ou_fit"]["max_iterations"] < 1:
        raise ValueError("Invalid OU fitting parameters")
    if s["ou_fit"]["tolerance"] <= 0 or len(s["ou_fit"]["initial_guess"]) != 3:
        raise ValueError("Invalid OU fitting tolerance or initial guess")
    if min(s["regression"]["ridge_alpha"], s["regression"]["ridge_alpha_per_sample"]) < 0:
        raise ValueError("Ridge penalties must be nonnegative")
    if s["regression"]["feature_order"] != ["global_policy_fitness", "number_contributing_agents", "gravity", "intercept"]:
        raise ValueError("Xi feature order cannot be changed")
    if s["regression"]["target_order"] != ["initialization", "log_mean_reversion_rate", "log_long_term_cumulative_reward", "log_sigma"]:
        raise ValueError("Xi target order cannot be changed")
    for key in ("output_matrix_file", "output_data_file"):
        if Path(s[key]).name != s[key]:
            raise ValueError("Output filenames must stay in Compression coexistence")
    if Path(s["output_matrix_file"]).suffix != ".json" or Path(s["output_data_file"]).suffix != ".pkl":
        raise ValueError("Matrix output must be .json and diagnostic output .pkl")
    if s["output_matrix_file"] == "setup calibration quantized.json":
        raise ValueError("Matrix output cannot overwrite the calibration setup")
    if not (HERE / q["codec_file"]).is_file():
        raise FileNotFoundError(q["codec_file"])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dependencies(s):
    global np, torch, gym, tqdm, core, codec, calibration
    import numpy as np
    import torch
    import gymnasium as gym
    from tqdm import tqdm
    sys.path.insert(0, str(HERE.parent / "Cosmic Octopi"))
    import utils as core
    device = s["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    core.device = torch.device(device)
    codec = load_module("compression_calibration_codec", HERE / s["quantization"]["codec_file"])
    calibration = load_module("compression_xi_fitting", HERE.parent / "Model calibration" / "script.py")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(spec, gravity, seed):
    env = gym.make(spec["id"], **spec["kwargs"])
    env.unwrapped.model.opt.gravity[:] = [0., 0., -gravity]
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def train_path(s, class_name, gravity, model, episodes, seed, label):
    """Fresh optimizer and rollout state for each independent subset branch."""
    seed_everything(seed)
    spec, p = s["environments"][class_name], s["training"]
    env = evaluation_env = None
    try:
        env = make_env(spec, gravity, seed)
        # Evaluation must not reset the training runner's environment mid-rollout.
        evaluation_env = make_env(spec, gravity, seed + 1)
        policy = core.Policy(copy.deepcopy(model))
        optimizer = torch.optim.Adam(policy.model.parameters(), lr=p["learning_rate"], eps=p["adam_eps"])
        runner = core.EnvRunner(env, policy, nsteps=p["rollout_steps"],
                               transforms=[core.GAE(policy, gamma=p["gamma"], lam=p["gae_lambda"])])
        sampler = core.TrajectorySampler(runner, num_epochs=p["sampler_epochs"],
                                        minibatch_size=p["minibatch_size"],
                                        transforms=[core.NormalizeAdvantages()])
        ppo = core.PPO(policy, optimizer, cliprange=p["cliprange"],
                       vf_coef=p["value_loss_coefficient"], max_grad_norm=p["max_grad_norm"])

        def evaluate():
            returns = []
            for _ in range(p["evaluation_episodes"]):
                obs, _ = evaluation_env.reset()
                total, done = 0.0, False
                while not done:
                    obs, reward, terminated, truncated, _ = evaluation_env.step(policy.act(obs)["actions"])
                    total += float(reward)
                    done = terminated or truncated
                returns.append(total)
            value = float(np.mean(returns))
            if not np.isfinite(value):
                raise ValueError("Non-finite measured evaluation return")
            return value

        path = [evaluate()]
        for _ in tqdm(range(episodes), desc=label, unit="cal. episode", leave=False):
            for _ in range(p["updates_per_episode"]):
                loss, grad = ppo.step(sampler.get_next())
                if not np.isfinite(loss) or not np.isfinite(grad):
                    raise ValueError("PPO training diverged; refusing to fit invalid rewards")
            path.append(evaluate())
        return np.asarray(path), copy.deepcopy(policy.model).cpu()
    finally:
        if env is not None:
            env.close()
        if evaluation_env is not None:
            evaluation_env.close()


def collect_wet_experiment(s, class_name, experiment):
    """Two warmups, then three independent quantized subset continuations."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [s["seed"], CLASS_NAMES.index(class_name), experiment]))
    fresh_seed = lambda: int(rng.integers(0, 2**31 - 2))
    n = s["n_agents"]
    gravities = rng.gamma(s["gravity_mean"]**2 / s["gravity_variance"],
                          s["gravity_variance"] / s["gravity_mean"], n)
    lo, hi = s["warmup_episode_bounds"]
    maximum = int(rng.integers(lo, hi + 1))
    warmups = rng.integers(lo, maximum + 1, n)
    seed_everything(fresh_seed())
    probe = make_env(s["environments"][class_name], float(gravities[0]), fresh_seed())
    try:
        initial_model = core.Network(probe.observation_space.shape[0], probe.action_space.shape[0],
                                     s["environments"][class_name]["hidden_size"])
    finally:
        probe.close()
    # All local agents share the same initialization, as required for FedAvg.
    paths, uploads, fitnesses = [], [], []
    for agent in range(n):
        path, local = train_path(s, class_name, float(gravities[agent]), initial_model,
                                 int(warmups[agent]), fresh_seed(),
                                 f"{class_name} wet {experiment + 1}: warmup {agent + 1}/{n}")
        paths.append(path)
        fitnesses.append(float(path[1:].mean()))
        uploads.append(codec.quantize_model(local))
    rows = []
    for count in range(1, n + 1):
        for subset in combinations(range(n), count):
            # Exactly the same int8 decode -> episode-weighted FedAvg as Oracle.
            decoded = [codec.dequantize_model(uploads[i]) for i in subset]
            weights = [int(warmups[i]) for i in subset]
            aggregate = core.federated_average(decoded, weights)
            target = int(rng.choice(subset))
            gravity = float(gravities[target])
            fitness = float(np.mean([fitnesses[i] for i in subset]))
            path, _ = train_path(s, class_name, gravity, aggregate, s["n_episodes"],
                                 fresh_seed(), f"{class_name} wet {experiment + 1}: subset {subset}")
            overhead = sum(codec.get_payload_size_in_kb(uploads[i]) for i in subset)
            unquantized = sum(core.get_model_size_in_kb(m) for m in decoded)
            if overhead >= unquantized:
                raise ValueError("Upload quantization did not reduce measured overhead")
            rows.append({"features": [fitness, count, gravity, 1.0], "reward_path": path,
                         "subset": subset, "target_agent": target,
                         "overhead_kib": overhead, "unquantized_overhead_kib": unquantized})
    return {"wet_experiment": experiment, "agent_gravities": gravities,
            "warmup_episodes": warmups, "warmup_reward_paths": paths,
            "local_fitnesses": fitnesses, "rows": rows}


def save_data(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def fit_matrices(s, data):
    fit = s["ou_fit"]
    bounds = [fit["bounds"][name] for name in
              ("mean_reversion_rate", "long_term_cumulative_reward", "sigma")]
    matrices, diagnostics = {}, {}
    for class_name in s["class_names"]:
        rows = [row for wet in data["classes"][class_name] for row in wet["rows"]]
        targets, valid, fits = [], [], []
        for row in tqdm(rows, desc=f"Fit {class_name} OU paths"):
            path = row["reward_path"]
            minimizer = calibration.RecordingMinimizer(method=fit["method"],
                            tol=fit["tolerance"], options={"maxiter": fit["max_iterations"]})
            model = calibration.OrnsteinUhlenbeck()
            estimator = calibration.AnalyticalMLE(sample=path, param_bounds=bounds,
                            dt=fit["dt"], density=calibration.ExactDensity(model), minimizer=minimizer)
            guess = np.clip(fit["initial_guess"], np.array(bounds)[:, 0], np.array(bounds)[:, 1])
            with redirect_stdout(io.StringIO()):
                estimate = estimator.estimate_params(guess)
            parameters = np.asarray(estimate.params)
            converged = bool(minimizer.last_result.success)
            ok = bool(np.isfinite(parameters).all() and (parameters > 0).all()
                      and np.isfinite(estimate.log_like)
                      and (converged or not fit["require_convergence"]))
            targets.append([path[0], *np.log(np.maximum(parameters, np.finfo(float).tiny))])
            valid.append(ok)
            fits.append({"parameters": parameters, "converged": converged,
                         "log_likelihood": estimate.log_like,
                         "at_bound": np.isclose(parameters, np.array(bounds)[:, 0]) |
                                     np.isclose(parameters, np.array(bounds)[:, 1])})
        design = np.asarray([r["features"] for r in rows])[valid]
        diagnostics[class_name] = {"fits": fits, "valid": valid,
                                   "rank": int(np.linalg.matrix_rank(design)) if len(design) else 0}
        data["fit_diagnostics"] = diagnostics
        save_data(HERE / s["output_data_file"], data)
        if len(design) < 4 or np.linalg.matrix_rank(design) < 4:
            raise ValueError(f"{class_name}: fewer than four independent valid feature rows; "
                             "raw data saved. Inspect fits before rerunning training.")
        alpha = max(s["regression"]["ridge_alpha"],
                    s["regression"]["ridge_alpha_per_sample"] * len(design))
        matrix = calibration.regress_matrix_regularized(design, np.asarray(targets)[valid], alpha)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{class_name}: non-finite fitted matrix")
        matrices[class_name] = matrix.tolist()
        diagnostics[class_name].update({"ridge_alpha": alpha,
                                        "condition_number": float(np.linalg.cond(design))})
        print(f"{class_name}: fitted Xi from {sum(valid)}/{len(rows)} valid paths", flush=True)
    return matrices, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=HERE / "setup calibration quantized.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate setup only; no simulation or output")
    mode.add_argument("--resume", action="store_true", help="Continue the saved complete wet experiments")
    mode.add_argument("--fit-only", action="store_true", help="Refit saved complete reward traces; no training")
    args = parser.parse_args()
    s = json.loads(args.setup.read_text(encoding="utf-8"))
    validate_setup(s)
    n_paths = s["n_wet_experiments"] * (2**s["n_agents"] - 1)
    print(f"{s['class_names']}: {s['n_agents']} agents, {s['n_wet_experiments']} wet experiments "
          f"per class, {n_paths} post-aggregation paths/class, T={s['n_episodes']}", flush=True)
    if args.check:
        print("Setup valid. No simulation was started and no results were written.")
        return
    output = HERE / s["output_data_file"]
    matrix_output = HERE / s["output_matrix_file"]
    if not (args.resume or args.fit_only) and (output.exists() or matrix_output.exists()):
        raise FileExistsError("Calibration output already exists; use --resume or --fit-only")
    load_dependencies(s)
    if args.resume or args.fit_only:
        with output.open("rb") as stream:
            data = pickle.load(stream)
        # Fitting settings may change without repeating wet experiments.
        ignored = {"ou_fit", "regression", "output_matrix_file", "output_data_file"}
        if {k: v for k, v in data["setup"].items() if k not in ignored} != {k: v for k, v in s.items() if k not in ignored}:
            raise ValueError("Saved wet experiment settings do not match the requested setup")
        data["setup"] = s
    else:
        data = {"setup": s, "classes": {c: [] for c in s["class_names"]},
                "complete": False, "reward_source": "measured_PPO_after_int8_upload"}
    if not args.fit_only:
        for class_name in s["class_names"]:
            for wet in tqdm(range(len(data["classes"][class_name]), s["n_wet_experiments"]),
                            desc=f"{class_name} wet experiments", unit="experiment"):
                data["classes"][class_name].append(collect_wet_experiment(s, class_name, wet))
                save_data(output, data)
    if any(len(data["classes"][c]) != s["n_wet_experiments"] for c in s["class_names"]):
        raise ValueError("Wet data are incomplete; use --resume before fitting")
    data["complete"] = True
    save_data(output, data)
    matrices, diagnostics = fit_matrices(s, data)
    data["fitted_Xi_matrix"], data["fit_diagnostics"] = matrices, diagnostics
    save_data(output, data)
    temporary = matrix_output.with_name(matrix_output.name + ".tmp")
    temporary.write_text(json.dumps(matrices, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(matrix_output)
    print(f"Saved calibrated matrices: {matrix_output}", flush=True)


if __name__ == "__main__":
    main()
