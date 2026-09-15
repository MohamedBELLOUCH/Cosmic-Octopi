"""Run the shared holistic comparison with MORBO, separately for RTS and STR."""

import sys

sys.dont_write_bytecode = True

from holistic_model_based_runner_quantized import main


if __name__ == "__main__":
    main("MORBO", __file__)
