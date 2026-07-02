## ml_pipeline/run_pipeline.py


import argparse

from src.preprocess import run_preprocessing
from src.train import run_training
from src.evaluate import run_evaluation
from src.export_tflite import run_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Women safety gesture image-only pipeline")
    parser.add_argument("command", choices=["preprocess", "train", "evaluate", "export"])
    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocessing()
    elif args.command == "train":
         run_training()
    elif args.command == "evaluate":
        run_evaluation()
    elif args.command == "export":
        run_export()


if __name__ == "__main__":
    main()