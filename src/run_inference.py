"""Run TrustMed E5 inference on preprocessed waveform windows.

Input shape: (N, 3, 3000)
Channel order: ART, ECG-II, PLETH.

Input must already follow the exact preprocessing and normalization
used during training. This script does not preprocess raw signals.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e5_model_architecture import TrustMedE5ResNetBiLSTM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npy", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/e5_best.pt")
    parser.add_argument("--output-csv", default="trustmed_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        default="auto")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    input_path = Path(args.input_npy)
    checkpoint_path = Path(args.checkpoint)

    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    windows = np.load(input_path, mmap_mode="r")

    if windows.ndim != 3 or tuple(windows.shape[1:]) != (3, 3000):
        raise SystemExit(
            f"Expected shape (N, 3, 3000); got {windows.shape}"
        )
    if len(windows) == 0:
        raise SystemExit("Input contains no windows.")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable.")
        device = torch.device(args.device)

    # Load only a trusted checkpoint created by this project.
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise SystemExit("Unexpected checkpoint format.")
    if "model_state_dict" not in checkpoint:
        raise SystemExit("Checkpoint lacks model_state_dict.")

    model = TrustMedE5ResNetBiLSTM()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    scores = []

    with torch.inference_mode():
        for start in range(0, len(windows), args.batch_size):
            end = min(start + args.batch_size, len(windows))
            array = np.array(windows[start:end], dtype=np.float32, copy=True)

            if not np.isfinite(array).all():
                raise SystemExit(
                    f"Non-finite input values in windows {start}:{end}. "
                    "Verify preprocessing before inference."
                )

            x = torch.from_numpy(array).to(device)
            logits = model(x)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            scores.extend(probabilities.tolist())

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "window_index": np.arange(len(scores)),
        "risk_score": scores,
    }).to_csv(output_path, index=False)

    print("Inference completed.")
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Device:", device)
    print("Windows scored:", len(scores))
    print("Output:", output_path.resolve())
    print(
        "Risk scores are model outputs, not verified calibrated "
        "clinical probabilities."
    )


if __name__ == "__main__":
    main()
