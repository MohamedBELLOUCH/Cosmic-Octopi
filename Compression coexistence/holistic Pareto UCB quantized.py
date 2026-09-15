"""Train the holistic Pareto UCB arm recommendations; do not evaluate held-out costs."""
import sys
sys.dont_write_bytecode = True
from holistic_runner_quantized import main

if __name__ == "__main__":
    main('Pareto UCB')
