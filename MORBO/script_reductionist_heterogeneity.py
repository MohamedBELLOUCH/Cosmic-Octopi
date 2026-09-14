"""Run or resume the reductionist MORBO request heterogeneity experiment."""

from heterogeneity_runner import main


if __name__ == "__main__":
    main("reductionist", "setup_reductionist_heterogeneity.json", __file__)
