import encoder
import activation
import torch

class Olfactor(torch.nn.Module):
    def __init__(self, encoder: encoder.Encoder, notes_dim: int, **kwargs):
        super(Olfactor, self).__init__()
        self.encoder = encoder
        self.embed_dim = self.encoder.readout.in_channels
        self.notes_dim = notes_dim
        self.predict = torch.nn.Linear(self.embed_dim,self.notes_dim)

    def forward(self,graph):
        embed = self.encoder(graph)
        logits = self.predict(embed)
        return {"embed":embed,"logits":logits}