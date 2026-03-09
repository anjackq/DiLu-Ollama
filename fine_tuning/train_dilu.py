"""
Legacy entrypoint kept for backward compatibility.
Use `fine_tuning/train_dilu_updated.py` for new options.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fine_tuning.train_dilu_updated import parse_args, train


if __name__ == "__main__":
    train(parse_args())
