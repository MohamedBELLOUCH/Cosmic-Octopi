"""Launch quantized MORBO/qNParEGO and random-threshold baseline comparisons."""
import argparse
import importlib
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=("both", "holistic", "reductionist"), default="both")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    from model_based_quantized import load_setup, payload_measurements
    modes = ("holistic", "reductionist") if args.approach == "both" else (args.approach,)
    # Validate all requested settings before any optimization or simulation.
    for mode in modes:
        setup = load_setup(HERE / f"setup_model_based_{mode}_quantized.json")
        runner = importlib.import_module(f"{mode}_model_based_runner_quantized")
        runner.validate_setup(setup)
        measurements = payload_measurements(setup)
        print(f"{mode}: horizon {setup[mode]['horizon']}; reference {setup['reference_costs']}; "
              f"{setup['baseline_sampling']['n_sequences']} random baseline sequences", flush=True)
        for name, measurement in measurements.items():
            print(f"  {name}: {measurement['float32_kib']:.3f} -> {measurement['quantized_kib']:.3f} KiB", flush=True)
        for algorithm in ("MORBO", "qNParEGO"):
            runner.load_backends(algorithm)  # Imports only; no optimization.
            hp = setup["hyperparameters"][algorithm]
            print(f"  {algorithm}: {hp['n_initial_points']} initial + {hp['n_iterations']} steps "
                  f"x {hp['batch_size']} candidates; RTS and STR", flush=True)
        command = [sys.executable, "-B", str(HERE / f"{mode} single fitness margin baseline quantized.py"), "--validate-only"]
        subprocess.run(command, check=True)
    if args.validate_only:
        print("Validation passed; no simulations, optimization or result files created.", flush=True)
        return
    from tqdm import tqdm
    with tqdm(total=4 * len(modes), desc="Quantized model-based comparisons", unit="stage", ascii=True) as bar:
        for mode in modes:
            scripts = [f"{mode} single fitness margin baseline quantized.py",
                       f"{mode} MORBO quantized.py", f"{mode} qNParEGO quantized.py",
                       f"evaluate_{mode}_model_based_comparison_quantized.py"]
            for index, filename in enumerate(scripts):
                bar.set_postfix_str(filename)
                command = [sys.executable, "-B", str(HERE / filename)]
                if index == 3:
                    command += ["--workers", str(args.workers)]
                subprocess.run(command, check=True)
                bar.update(1)
    print("Complete. Quantized model-based comparison files are saved in Compression coexistence.", flush=True)


if __name__ == "__main__":
    main()
