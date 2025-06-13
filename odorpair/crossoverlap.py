import torch
import utils

from odorpair import activation, olfactor


class CrossEncoder(torch.nn.Module):

    def __init__(self, olfactor: olfactor.Olfactor, representation_mode: str,
                 cross_encoder_dim: int, act_mode: str, do_encoder_diff: bool,
                 do_sigmoid: bool, cross_encoder_layers: int, **kwargs):
        super(CrossEncoder, self).__init__()
        self.olfactor = olfactor
        self.representation_mode = representation_mode
        if self.representation_mode == "embed":
            self.input_dim = self.olfactor.embed_dim
        if self.representation_mode == "notes":
            self.input_dim = self.olfactor.notes_dim
        if self.representation_mode == "both":
            self.input_dim = self.olfactor.embed_dim + self.olfactor.notes_dim

        self.do_encoder_diff = do_encoder_diff
        if self.do_encoder_diff:
            combined_dim = self.input_dim
        else:
            combined_dim = 2 * self.input_dim

        act_fn = activation.get_act_fn(act_mode)

        self.do_sigmoid = do_sigmoid
        self.readout = utils.build_layers(
            in_dim=combined_dim,
            hidden_dim=cross_encoder_dim,
            out_dim=1,
            act_fn=act_fn,
            num_hidden_layers=cross_encoder_layers,
        )
        if cross_encoder_dim > 0:
            self.readout = torch.nn.Sequential(
                torch.nn.Linear(combined_dim, cross_encoder_dim),
                act_fn(),
                torch.nn.Linear(cross_encoder_dim, 1),
            )
        else:
            self.readout = torch.nn.Linear(combined_dim, 1)

        self.overlap = torch.nn.Linear(2, 1)

    def get_representation(self, graph):
        olf = self.olfactor(graph)
        if self.representation_mode == "embed":
            return olf["embed"]
        if self.representation_mode == "notes":
            return olf["logits"]
        return torch.cat([olf["embed"], olf["logits"]], dim=-1)

    def forward(self, blend):
        repr1 = self.get_representation(blend["graph1"])
        repr2 = self.get_representation(blend["graph2"])
        if self.do_encoder_diff:
            x = (repr1 - repr2).square()
        else:
            x = torch.cat([repr1, repr2], dim=-1)

        x = self.readout(x)
        pred = torch.nn.functional.sigmoid(x) if self.do_sigmoid else x
        blend = blend["overlap"].unsqueeze(-1)
        return self.overlap(torch.cat([pred, blend], dim=-1)).squeeze()
