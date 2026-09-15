"""Run the Cheetah reductionist comparison with MORBO, separately for RTS and STR."""

import sys

sys.dont_write_bytecode = True

from reductionist_model_based_runner import main


if __name__ == "__main__":
    main("MORBO", __file__)
