import gc
import time

import numpy as np
import optuna
import torch
import torch_geometric as pyg
import torchmetrics
from tqdm.notebook import tqdm

from odorpair import gcn, storage, utils


def predict(graph_model, batch):
    batch = {k: v.cuda() for k, v in batch.items() if k != "pair"}
    output = graph_model(batch["graph"])
    return output["logits"], batch["notes"]


def to_np(tsr):
    return tsr.cpu().detach().numpy()


def do_train_step(optimizer, scheduler, graph_model, batch):
    optimizer.zero_grad()

    logits, targets = predict(graph_model, batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets)

    loss.backward()

    optimizer.step()
    scheduler.step()

    return to_np(loss)


def calc_test_metrics(graph_model, config, test_loader):
    tls = []
    # For per-note AUROC
    auroc_metric_per_note = torchmetrics.classification.MultilabelAUROC(
        config["notes_dim"], average=None)
    # For micro-average AUROC
    auroc_metric_micro = torchmetrics.classification.MultilabelAUROC(
        config["notes_dim"], average="micro")

    with torch.no_grad():
        for batch in test_loader:
            logits, targets = predict(graph_model, batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets)

            auroc_metric_per_note.update(logits, targets.int())
            auroc_metric_micro.update(logits, targets.int())

            tls.append(to_np(loss))

    auroc_per_note = to_np(auroc_metric_per_note.compute())  # Per-note AUROC
    auroc_micro = to_np(auroc_metric_micro.compute())  # Micro-average AUROC

    return {
        "auroc": auroc_micro,  # Micro-average AUROC
        "loss": np.mean(tls),
        "auroc_per_note": dict(zip(config["covered_notes"], auroc_per_note)),
    }


def train_model(graph_model, config, train_dataset, test_dataset, trial=None):
    # Nested utils
    def make_ckpt_data():
        elapsed = time.perf_counter() - start
        auroc = test_aurocs[-1] if len(test_aurocs) > 0 else 0

        data = config | {
            "train_loss": train_losses,
            "test_loss": test_losses,
            "test_aurocs": test_aurocs,
            "time": elapsed,
            "auroc": auroc,
        }
        data["count_parameters"] = model_size
        return utils.make_serializable(data)

    start = time.perf_counter()

    model_size = utils.count_parameters(graph_model)

    print(
        f"Trial ID = {config['trial_id']}",
        f"Model Size = {model_size:,}",
        config,
        sep="\n",
    )
    if not trial is None and (model_size <= config["max_size"]
                              or model_size >= config["min_size"]):
        print(f"Invalid Size {model_size:,}")
        raise optuna.TrialPruned()

    graph_model.cuda()

    train_loader = pyg.loader.DataLoader(train_dataset,
                                         batch_size=config["bsz"],
                                         shuffle=True)
    test_loader = pyg.loader.DataLoader(test_dataset,
                                        batch_size=config["bsz"],
                                        shuffle=True)

    optimizer = torch.optim.AdamW(
        graph_model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
    )

    total_steps = config["epochs"] * len(train_loader)
    scheduler = utils.make_scheduler(optimizer, config["warmup"], total_steps)

    step = 0
    test_losses = []
    test_aurocs = []
    train_losses = []

    test_metrics = calc_test_metrics(graph_model, config, test_loader)

    with tqdm(total=total_steps, smoothing=0) as pbar:
        for epoch in range(config["epochs"]):
            # Do training loop
            graph_model.train()
            btls = []

            for batch in train_loader:
                loss = do_train_step(optimizer, scheduler, graph_model, batch)
                btls.append(loss)

                step += 1
                pbar.update(1)

            train_losses.append(np.mean(btls).item())

            # Evaluate graph_model on training data
            graph_model.eval()
            test_metrics = calc_test_metrics(graph_model, config, test_loader)

            test_losses.append(test_metrics["loss"].item())
            test_aurocs.append(test_metrics["auroc"].item())

            if not trial is None:
                # Early stopping
                trial.report(test_metrics["auroc"], epoch)
                if trial.should_prune():
                    print(
                        f"Pruning after {epoch} epochs w/ score {best_auroc:.2f}"
                    )
                    storage.save(config["trial_path"])
                    raise optuna.TrialPruned()
            else:
                print(epoch, test_metrics)
                print(
                    "NOTES:",
                    len([
                        k for k, v in test_metrics["auroc_per_note"].items()
                        if v > 0.5
                    ]),
                )

    if not trial is None:
        storage.save(config["trial_path"])

    print({k: v for k, v in test_metrics["auroc_per_note"].items() if v > 0.5})

    return graph_model, max(test_aurocs)
