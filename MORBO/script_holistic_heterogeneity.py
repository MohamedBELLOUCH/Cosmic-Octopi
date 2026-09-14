"""Run or resume the holistic MORBO request heterogeneity experiment."""

from heterogeneity_runner import main


if __name__ == "__main__":
    main("holistic", "setup_holistic_heterogeneity.json", __file__)
