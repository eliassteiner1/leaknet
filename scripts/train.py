"""
Runs one (or several) of the experiments with the settings from leaknet.train.EXPERIMENTS and saves the loss history
and the weights to results/.

    uv run scripts/train.py single_leak
    uv run scripts/train.py sinkhorn blobs conv_heatmap --seed 0
"""

import argparse

from leaknet.train import EXPERIMENTS, run_experiment, save_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", choices=list(EXPERIMENTS))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default=None, help="default: cuda:0 if available, else cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for name in args.experiments:
        print(f"\n### {name}: {EXPERIMENTS[name]['note']}")
        network, history = run_experiment(name, data_dir=args.data_dir, device=args.device, seed=args.seed)
        save_run(network, history, args.out_dir)
        print(f"best validation loss: {history['valid_best']:.4f} | {history['samples_per_second']:.0f} samples/s | {history['train_time_s']:.0f}s")
