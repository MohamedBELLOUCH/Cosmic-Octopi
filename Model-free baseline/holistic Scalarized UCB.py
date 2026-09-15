"""Train the holistic Scalarized UCB arm recommendations; do not evaluate held-out costs."""
import sys
sys.dont_write_bytecode = True
from holistic_runner import main

if __name__ == "__main__":
    main('Scalarized UCB')
