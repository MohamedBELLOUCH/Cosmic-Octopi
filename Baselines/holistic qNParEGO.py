"""Run the shared holistic comparison with qNParEGO, separately for RTS and STR."""

import sys

sys.dont_write_bytecode = True

from holistic_model_based_runner import main


if __name__ == "__main__":
    main("qNParEGO", __file__)
