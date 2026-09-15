"""Run both quantized model-free comparisons sequentially, with progress bars."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=("both", "holistic", "reductionist"), default="both")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("workers must be positive")
    modes = ("holistic", "reductionist") if args.approach == "both" else (args.approach,)
    folder = Path(__file__).resolve().parent
    for mode in modes:
        command = [sys.executable, "-B", str(folder / f"run_{mode}_comparison_quantized.py")]
        if args.workers is not None:
            command.extend(["--workers", str(args.workers)])
        if args.validate_only:
            command.append("--validate-only")
        print(f"Quantized model-free comparison: {mode}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
