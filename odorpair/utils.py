import torch


def count_parameters(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters())


def readout_counts(module: torch.nn.Module):
    results = {"total": count_parameters(module)}
    for n, c in module.named_children():
        results[n] = count_parameters(c)
    return results


def make_scheduler(optimizer, warmup, total_steps):
    warmup_steps = int(warmup * total_steps)
    cooldown_steps = total_steps - warmup_steps
    # Define the warmup and cooldown schedulers
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-10,
        end_factor=1.0,
        total_iters=warmup_steps)
    cooldown_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1e-10,
        total_iters=cooldown_steps)
    # Chain the schedulers together with SequentialLR
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cooldown_scheduler],
        milestones=[warmup_steps])


def build_layers(in_dim, hidden_dim, out_dim, act_fn, num_hidden_layers):
    layers = []

    # If there are no hidden layers, just add the output layer
    if num_hidden_layers == 0:
        layers.append(torch.nn.Linear(in_dim, out_dim))
    else:
        # Input layer
        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        layers.append(act_fn())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())

        # Output layer
        layers.append(torch.nn.Linear(hidden_dim, out_dim))

    return torch.nn.Sequential(*layers)
