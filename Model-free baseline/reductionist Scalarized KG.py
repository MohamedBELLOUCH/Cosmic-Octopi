"""Train the reductionist Scalarized KG arm recommendations; do not evaluate held-out costs."""
import sys
sys.dont_write_bytecode = True
from reductionist_runner import main

if __name__ == "__main__":
    main('Scalarized KG')
