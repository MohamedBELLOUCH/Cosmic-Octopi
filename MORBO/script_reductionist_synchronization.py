"""Run the reductionist MORBO synchronization experiment from its JSON setup."""

from synchronization_runner import main


if __name__ == "__main__":
    main("reductionist", "setup_reductionist_synchronization.json", __file__)
