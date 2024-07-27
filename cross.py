import olfactor
import activation
import torch

class CrossEncoder(torch.nn.Module):
    def __init__(self,olfactor:olfactor.Olfactor,representation_mode: str, cross_encoder_dim: int, act_mode: str, do_encoder_diff: bool,**kwargs):
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
          combined_dim = 2*self.input_dim

        act_fn = activation.get_act_fn(act_mode)
        if cross_encoder_dim > 0:
          self.readout = torch.nn.Sequential(torch.nn.Linear(combined_dim,cross_encoder_dim),act_fn(),
                                            torch.nn.Linear(cross_encoder_dim,1))
        else:
          self.readout = torch.nn.Linear(combined_dim,1)

    def get_representation(self,graph):
      olf = self.olfactor(graph)
      if self.representation_mode == "embed":
        return olf["embed"]
      if self.representation_mode == "notes":
        return olf["logits"]
      return torch.cat([olf["embed"],olf["logits"]],dim=-1)

    def forward(self,graph1, graph2):
      repr1 = self.get_representation(graph1)
      repr2 = self.get_representation(graph2)
      if self.do_encoder_diff:
        x = (repr1-repr2).square()
      else:
        x = torch.cat([repr1,repr2],dim=-1)

      return self.readout(x).squeeze(dim=-1)