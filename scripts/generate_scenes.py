"""
Generates a collection of random scenes and writes them to an hdf5 file.

    uv run scripts/generate_scenes.py --n-samples 20000 --out data/single_leak_20k.h5 --seed 0
    uv run scripts/generate_scenes.py --n-samples 50000 --n-leak-max 3 --out data/multi_leak_50k.h5 --seed 1
"""

import argparse
import os
from pathlib import Path

from leaknet.simulation import default_filename, generate_collection, write_h5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-timesteps", type=int, default=100)
    parser.add_argument("--n-sensor-patterns", type=int, default=50, help="size of the fixed set of sensor layouts, 0 = new layout for every scene")
    parser.add_argument("--n-sensor-min", type=int, default=6)
    parser.add_argument("--n-sensor-max", type=int, default=10)
    parser.add_argument("--n-source-min", type=int, default=15)
    parser.add_argument("--n-source-max", type=int, default=30)
    parser.add_argument("--n-leak-min", type=int, default=1)
    parser.add_argument("--n-leak-max", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", type=Path, default=None, help="default: data/<timestamp>--X<n_samples>.h5")
    args = parser.parse_args()

    out = args.out or Path("data") / default_filename(args.n_samples)
    out.parent.mkdir(parents=True, exist_ok=True)

    collection = generate_collection(
        n_samples=args.n_samples,
        n_timesteps=args.n_timesteps,
        n_sensor_patterns=args.n_sensor_patterns,
        n_sensor_min=args.n_sensor_min,
        n_sensor_max=args.n_sensor_max,
        n_source_min=args.n_source_min,
        n_source_max=args.n_source_max,
        n_leak_min=args.n_leak_min,
        n_leak_max=args.n_leak_max,
        seed=args.seed,
        n_workers=args.workers,
    )
    write_h5(collection, out)
    print(f"wrote {len(collection)} scenes to {out}")
