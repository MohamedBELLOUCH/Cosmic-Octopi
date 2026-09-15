"""Train the three reductionist algorithms, then evaluate their final sets."""
import sys
sys.dont_write_bytecode = True
import argparse
from tqdm import tqdm
from reductionist_runner import load_setup, train
from evaluate_reductionist_comparison import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    setup = load_setup()
    workers = setup["evaluation"]["workers"] if args.workers is None else args.workers
    if workers < 1:
        parser.error("workers must be positive")
    if args.validate_only:
        print("Cheetah reductionist setup validated; no training, evaluation or result files created.")
        return
    algorithms = ("Pareto UCB", "Scalarized UCB", "Scalarized KG")
    with tqdm(total=4, desc="Reductionist comparison", unit="stage", ascii=True) as progress:
        for algorithm in algorithms:
            progress.set_postfix_str(algorithm)
            train(algorithm)
            progress.update(1)
        progress.set_postfix_str("held-out evaluation")
        evaluate(workers)
        progress.update(1)
    print("Complete. Run the reductionist section in Paper plots/Model-free baseline comparison.ipynb to display the saved results.")


if __name__ == "__main__":
    main()
