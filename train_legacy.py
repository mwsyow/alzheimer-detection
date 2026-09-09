"""Compatibility entrypoint for fixed-test, single-split, and refit runs."""

from train import main


if __name__ == "__main__":
    main(legacy_only=True)
