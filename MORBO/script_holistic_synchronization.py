"""Run the holistic MORBO synchronization experiment from its JSON setup."""

from synchronization_runner import main


if __name__ == "__main__":
    main("holistic", "setup_holistic_synchronization.json", __file__)
