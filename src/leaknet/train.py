"""
The training loop that all experiments share, and the list of experiments with the settings they were run with.
"""

import json
import math
import random
import time
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader, default_collate, random_split

from leaknet import losses
from leaknet.data import MyDataset, ScheduledBatchSampler, load_scenes
from leaknet.models import build_model

# the skewed std is nan (with a warning) for tiny batches, that is expected
warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom")

# batch size schedules: {batch size : how many epochs, ... : ...}, no duplicate keys!
BS_SCHEDULE_PHASE_1 = {32: 1, 64: 2, 128: 4, 256: 8, 512: 16, 1024: 32}
BS_SCHEDULE_PHASE_2 = {32: 1, 64: 2, 128: 4, 256: 8, 512: 16, 1024: 16}

EXPERIMENTS = {
    # phase 1 --------------------------------------------------------------------------------------------------------
    "single_leak": dict(
        note="phase 1: one leak, regress its position. loss = distance to leak",
        model="position",
        criterion=losses.distance_loss,
        data_file="single_leak_20k.h5",
        n_samples=None,
        train_fraction=0.8,
        bs_schedule=BS_SCHEDULE_PHASE_1,
        weight_decay=10**-5,
    ),
    "mlp_baseline": dict(
        note="phase 1: same task, but just flatten the whole input and pass it to an mlp",
        model="mlp",
        criterion=losses.distance_loss,
        data_file="single_leak_20k.h5",
        n_samples=None,
        train_fraction=0.8,
        bs_schedule=BS_SCHEDULE_PHASE_1,
        weight_decay=10**-5,
    ),
    # phase 2 --------------------------------------------------------------------------------------------------------
    "sinkhorn": dict(
        note="phase 2: n leaks, slots with confidence + soft permutation (sinkhorn) with FN & FP punishing",
        model="slots",
        criterion=losses.matching_loss,
        aux_names=("TP", "FN", "FP"),
        data_file="multi_leak_50k.h5",
        n_samples=None,
        train_fraction=0.9,
        bs_schedule=BS_SCHEDULE_PHASE_2,
        weight_decay=10**-5,
    ),
    "blobs": dict(
        note="phase 2: n leaks, confidence heatmap from parametric gaussian blobs (overfit test on 10 scenes)",
        model="blobs",
        criterion=losses.blob_heatmap_loss,
        data_file="multi_leak_50k.h5",
        n_samples=10,
        train_fraction=0.8,
        bs_schedule={32: 1000},
        weight_decay=0,
    ),
    "conv_heatmap": dict(
        note="phase 2: n leaks, confidence heatmap from convolution + upsampling (overfit test on 10 scenes)",
        model="conv_heatmap",
        criterion=losses.weighted_heatmap_loss,
        data_file="multi_leak_50k.h5",
        n_samples=10,
        train_fraction=0.8,
        bs_schedule={32: 1000},
        weight_decay=0,
    ),
    "linear_heatmap": dict(
        note="phase 2: n leaks, confidence heatmap direct from linear layer (overfit sanity check on 16 scenes)",
        model="linear_heatmap",
        criterion=losses.plain_heatmap_loss,
        data_file="multi_leak_50k.h5",
        n_samples=16,
        train_fraction=0.8,
        bs_schedule={32: 2000},
        weight_decay=0,
        skewed_std=False,
        diagnostics=True,
    ),
    "count": dict(
        note="phase 2: just predict the number of leaks (sanity check)",
        model="count",
        criterion=losses.count_loss,
        data_file="multi_leak_50k.h5",
        n_samples=1000,
        train_fraction=0.8,
        bs_schedule={32: 1000},
        weight_decay=0,
    ),
}


def pick_device(device=None):
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def to_device(sample, device):
    return {key: value.to(device) for key, value in sample.items()}


def train_valid_split(data, train_fraction=0.8, seed=None):
    """random split of a list of scenes. With the same seed this gives back the split that a run was trained with."""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    dataset = MyDataset(data)
    L = len(dataset)
    return random_split(dataset, [int(train_fraction * L), L - int(train_fraction * L)])


def loss_stats(loss, skewed_std=True):
    """
    mean of the per-sample losses plus a "skewed" standard deviation: one for all the losses above the mean and one
    for all below. (nan if there are less than two samples on one side)
    """
    avg = torch.mean(loss)

    if skewed_std:
        std_hi = torch.std(loss[loss > avg])
        std_lo = torch.std(loss[loss < avg])
    else:
        std_hi = std_lo = torch.std(loss)

    return avg, std_hi, std_lo


def do_loss(criterion, sample, pred, skewed_std=True):
    """
    convenience for calculating loss statistics (returns ATTACHED training loss stuff). If the criterion returns a
    tuple, the first element is the loss that is trained on and the rest are auxiliary losses that are only logged.
    """
    loss = criterion(sample, pred)
    aux = []
    if isinstance(loss, tuple):
        loss, aux = loss[0], [torch.mean(el).item() for el in loss[1:]]

    return *loss_stats(loss, skewed_std), aux


def do_validation(criterion, network, validation_dataloader, device, skewed_std=True):
    """function for calculating validation error (returns DETACHED validation loss stuff)"""
    network.eval()

    with torch.no_grad():
        loss_val = []
        for sample in validation_dataloader:
            sample = to_device(sample, device)
            loss = criterion(sample, network(sample))
            loss_val.append(loss[0] if isinstance(loss, tuple) else loss)

        avg, std_hi, std_lo = loss_stats(torch.cat(loss_val), skewed_std)

    return avg.item(), std_hi.item(), std_lo.item()


def optimizer_diagnostics(model, optim):
    """some optimizer stats, to see if anything moves at all when a model refuses to overfit"""
    grad_norm = sum(p.grad.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    weight_norm = sum(p.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5

    # parameter update magnitude (adam-specific)
    total_update = 0.0
    for param in model.parameters():
        if param.grad is not None:
            state = optim.state[param]
            if "exp_avg_sq" in state:
                update = param.grad / (torch.sqrt(state["exp_avg_sq"]) + optim.defaults["eps"])
                total_update += update.norm(2).item() ** 2

    return grad_norm, weight_norm, total_update**0.5


def train(
    data,
    model="position",
    criterion=losses.distance_loss,
    bs_schedule=BS_SCHEDULE_PHASE_1,
    train_fraction=0.8,
    lr=10**-3,
    weight_decay=10**-5,
    skewed_std=True,
    diagnostics=False,
    aux_names=(),
    device=None,
    seed=None,
    plotter=None,
    verbose=True,
    **_,
):
    """
    Trains one model and returns (model, history). history holds everything that the live plot shows, so that the
    curves can still be looked at afterwards.

    Args:
        data: list of scene dicts (see data.load_scenes)
        model: name of the head ("position", "slots", ...) or "mlp"
        criterion: one of the functions in losses.py
        bs_schedule: {batch size : how many epochs, ...}
        plotter: optional plotting.TrainingPlotter for a live plot in a notebook
    """
    device = pick_device(device)
    n_epochs = sum(bs_schedule.values())

    # load dataset and split
    ds_tra, ds_val = train_valid_split(data, train_fraction, seed)

    # setup dataloaders. Train loader is setup with the custom sampler to implement the batch size schedule
    sampler_train = ScheduledBatchSampler(ds_tra, bs_schedule, shuffle=True, drop_last=False)
    loader_train = DataLoader(ds_tra, batch_sampler=sampler_train)
    loader_valid = DataLoader(ds_val, batch_size=len(ds_val), shuffle=False)

    # initialize model and move to device
    network = build_model(model).to(device)
    optim = torch.optim.Adam(network.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    history = {
        "n_train": len(ds_tra),
        "n_valid": len(ds_val),
        "n_epochs": n_epochs,
        "bs_schedule": {str(key): value for key, value in bs_schedule.items()},
        "aux_names": list(aux_names),
        "train": {"samples_seen": [], "loss": [], "std_hi": [], "std_lo": [], "aux": []},
        "valid": {"samples_seen": [], "loss": [], "std_hi": [], "std_lo": []},
        "diagnostics": {"grad_norm": [], "weight_norm": [], "update_magnitude": []},
    }

    def log_validation(samples_seen):
        val_avg, val_std_hi, val_std_lo = do_validation(criterion, network, loader_valid, device, skewed_std)
        for key, value in zip(("samples_seen", "loss", "std_hi", "std_lo"), (samples_seen, val_avg, val_std_hi, val_std_lo)):
            history["valid"][key].append(value)
        if plotter is not None:
            plotter.update_val_err_plot(val_avg, val_std_hi, val_std_lo)
        return val_avg

    # Training loop ----------------------------------------------------------------------------------------------------
    # do an initial validation
    samples_seen = 0
    val_best = log_validation(samples_seen)
    start_time = time.time()

    for E in range(n_epochs):
        # activate gradient calculation
        network.train()
        # update the batch size scheduler
        sampler_train.set_epoch(E)

        for sample in loader_train:
            # get the actual batch size (use some quantity, where batch size can be easily seen)
            actual_bs = len(sample["n_sensors"])
            samples_seen += actual_bs
            sample = to_device(sample, device)
            batch_start_time = time.time()

            optim.zero_grad()
            pred = network(sample)
            tra_avg, tra_std_hi, tra_std_lo, aux = do_loss(criterion, sample, pred, skewed_std)
            tra_avg.backward()
            optim.step()

            batch_end_time = time.time()

            for key, value in zip(("samples_seen", "loss", "std_hi", "std_lo", "aux"), (samples_seen, tra_avg.item(), tra_std_hi.item(), tra_std_lo.item(), aux)):
                history["train"][key].append(value)
            if diagnostics:
                with torch.no_grad():
                    for key, value in zip(("grad_norm", "weight_norm", "update_magnitude"), optimizer_diagnostics(network, optim)):
                        history["diagnostics"][key].append(value)
            if plotter is not None:
                plotter.update_proc_speed(actual_bs, batch_start_time, batch_end_time)
                plotter.update_tra_err_plot(samples_seen, tra_avg.item(), tra_std_hi.item(), tra_std_lo.item(), aux)

        # validation at end of each epoch
        val_avg = log_validation(samples_seen)
        # check if the new validation loss is the new best one and update if true
        val_best = min(val_best, val_avg)

        if verbose and (E % max(1, n_epochs // 20) == 0 or E == n_epochs - 1):
            print(f"Ep: {E:4d} | train: {history['train']['loss'][-1]:.4f} | valid: {val_avg:.4f} | best valid: {val_best:.4f}")

        # empty GPU memory, not sure how much this helps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    history["valid_best"] = val_best
    history["train_time_s"] = time.time() - start_time
    history["samples_per_second"] = samples_seen / history["train_time_s"]

    return network, history


def run_experiment(name, data_dir="data", device=None, seed=None, plotter=None, verbose=True):
    """runs one of the EXPERIMENTS with its settings"""
    config = EXPERIMENTS[name]
    data = load_scenes(Path(data_dir) / config["data_file"], config["n_samples"])
    network, history = train(data, device=device, seed=seed, plotter=plotter, verbose=verbose, **config)
    history["experiment"] = name
    history["note"] = config["note"]
    history["data"] = config["data_file"]
    history["seed"] = seed

    return network, history


def predict(network, scenes, device="cpu", batch_size=500):
    """model output for a list (or Subset) of scenes, on the cpu"""
    network.to(device).eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(scenes), batch_size):
            batch = default_collate([scenes[i] for i in range(start, min(start + batch_size, len(scenes)))])
            preds.append(network(to_device(batch, device)).cpu())

    return torch.cat(preds)


def save_run(network, history, out_dir="results"):
    """history as json (nan -> null) and the weights next to it"""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    def clean(obj):
        if isinstance(obj, dict):
            return {key: clean(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(value) for value in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    name = history["experiment"]
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(clean(history), f)
    torch.save(network.state_dict(), out_dir / f"{name}.pt")


def load_run(name, out_dir="results", device="cpu"):
    """returns (model with the trained weights, history) of a saved run"""
    out_dir = Path(out_dir)
    with open(out_dir / f"{name}.json") as f:
        history = json.load(f)

    network = build_model(EXPERIMENTS[name]["model"])
    network.load_state_dict(torch.load(out_dir / f"{name}.pt", map_location=device))
    network.to(device).eval()

    return network, history
