import torch


def count_parameters(module: torch.nn.Module):
    return "{:,}".format(sum(p.numel() for p in module.parameters()))


def readout_counts(module: torch.nn.Module):
    results = {"total": count_parameters(module)}
    for n, c in module.named_children():
        results[n] = count_parameters(c)
    return results
