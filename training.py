import optuna
import gc
import time
from tqdm.notebook import tqdm
import torchmetrics
import numpy as np
import dream.gcn
import dream.utils
import torch_geometric as pyg
import torch

def predict(graph_model,batch):
    batch = {k:v.cuda() for k, v in batch.items() if k!= "pair"}
    output = graph_model(batch["graph"])
    return output["logits"], batch["notes"]

def to_np(tsr):
  return tsr.cpu().detach().numpy()

def do_train_step(optimizer, scheduler, graph_model, batch):
    optimizer.zero_grad()

    logits, targets = predict(graph_model,batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    loss.backward()

    optimizer.step()
    scheduler.step()

    return to_np(loss)

def calc_test_metrics(graph_model, config, test_loader):
    tls = []
    # For per-note AUROC
    auroc_metric_per_note = torchmetrics.classification.MultilabelAUROC(config["notes_dim"], average=None)
    # For micro-average AUROC
    auroc_metric_micro = torchmetrics.classification.MultilabelAUROC(config["notes_dim"], average='micro')

    with torch.no_grad():
        for batch in test_loader:
            logits, targets = predict(graph_model, batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

            auroc_metric_per_note.update(logits, targets.int())
            auroc_metric_micro.update(logits, targets.int())

            tls.append(to_np(loss))

    auroc_per_note = to_np(auroc_metric_per_note.compute())  # Per-note AUROC
    auroc_micro = to_np(auroc_metric_micro.compute())        # Micro-average AUROC

    return {
        "auroc": auroc_micro,  # Micro-average AUROC
        "loss": np.mean(tls),
        "auroc_per_note": dict(zip(config["covered_notes"],auroc_per_note))
    }

def train_model(graph_model, config, train_dataset, test_dataset):
  start = time.perf_counter()

  graph_model = dream.gcn.GCN(**config)
  model_size = dream.utils.count_parameters(graph_model)
  assert model_size <= config["max_size"]
  assert model_size >= config["min_size"]

  graph_model.cuda()
  print(f"Trial ID = {config['trial_id']}",f"Model Size = {model_size:,}",config,sep="\n")

  train_loader = pyg.loader.DataLoader(train_dataset, batch_size=config["bsz"], shuffle=True)
  test_loader = pyg.loader.DataLoader(test_dataset, batch_size=config["bsz"], shuffle=True)

  optimizer = torch.optim.AdamW(graph_model.parameters(),lr=config["lr"], weight_decay=config["weight_decay"], betas=config["betas"])

  total_steps = config["epochs"]*len(train_loader)
  scheduler = dream.utils.make_scheduler(optimizer,config["warmup"],total_steps)

  step = 0
  test_losses = []
  test_aurocs = []
  train_losses = []

  test_metrics = calc_test_metrics(graph_model, config, test_loader)

  with tqdm(total=total_steps,smoothing=0) as pbar:
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
        print(epoch, test_metrics)
        print("NOTES:",len([k for k,v in test_metrics["auroc_per_note"].items() if v > .5]))
        test_losses.append(test_metrics["loss"].item())
        test_aurocs.append(test_metrics["auroc"].item())

  return graph_model, max(test_aurocs)