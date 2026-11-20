"""The two handwritten attention building blocks (single head, optional key masking)."""

import torch
import torch.nn as nn


# basic attention block with optional masking
class SelfAttention(nn.Module):
    def __init__(self, input_feats_dim, latent_dim, output_feats_dim) -> None:
        """
        input_feats are the main data. latent_dim can be chosen.
        -> note, needs skip connection on the outside!
        -> note, this is modified from standard. Normally input_feats_dim = output_feats_dim!
        """
        super().__init__()

        self.input_feats_dim = input_feats_dim
        self.latent_dim = latent_dim
        self.output_feats_dim = output_feats_dim

        # define attention weights (= matrices Wq, Wk, Wv)
        self.query = nn.Linear(self.input_feats_dim, self.latent_dim)
        self.key = nn.Linear(self.input_feats_dim, self.latent_dim)
        self.value = nn.Linear(self.input_feats_dim, self.latent_dim)

        # define last linear reshaping layer (= matrix Wout)
        self.linear_out = nn.Linear(self.latent_dim, self.output_feats_dim)

        # softmax (row-wise!)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, msk=None):
        """
        input x  : [BS (var, broadcastable) | input length (var) | feats (known, architecture defining)]
        input msk: [BS (var, broadcastable) | input length (var) ]   (1 = ignore this element)
        """

        # fallback for when no mask is provided
        if msk is None:
            msk = torch.zeros(x.shape[0], x.shape[1], device=x.device)

        # calculate the embedded vectors in latent space (= matrices Q, K, V)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # calculate the attention matrix -> normalize by latent dimension -> apply softmax -> and weigh it with V
        # mask has [BS, maskvec]. needs unsqueeze(1) so that it is a [BS, 1, maskvec] "lying" row vector -> broadcasted
        SCO = torch.bmm(Q, K.transpose(1, 2)) / (self.latent_dim**0.5)
        SCO = SCO.masked_fill(msk.unsqueeze(1) == 1, float("-inf"))
        ATT = self.softmax(SCO)
        WGH = torch.bmm(ATT, V)

        # apply last linear layer to reshape output
        x = self.linear_out(WGH)

        return x


# basic attention pooling block with optional masking
class AttentionPooling(nn.Module):
    def __init__(self, input_feats_dim, hidden_layers=[128]) -> None:
        """
        Implements the attention pooling mechanism. A simple mlp learns one weight per input element. The output is
        then a linear combination of all the weights and inputs.
        """
        super().__init__()

        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # set up the sequential mlp to learn the per-input weights [BS | input length | 1]
        layers = []
        current_feats = input_feats_dim
        for el in hidden_layers:
            layers.append(nn.Linear(current_feats, el))
            layers.append(self.activation)
            current_feats = el
        layers.append(nn.Linear(current_feats, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, msk=None):
        """
        input x:   [BS (var, broadcastable) | input length (var) | n feats (known, architecture defining)]
        input msk: [BS (var, broadcastable) | input length (var) ]   (1 = ignore this element)
        output:    [BS | 1 | n feats]
        """

        if msk is None:
            msk = torch.zeros(x.shape[0], x.shape[1], device=x.device)

        # calculate the weights by learning them with an mlp
        d = self.mlp(x)
        d = d.masked_fill(msk.unsqueeze(2) == 1, float("-inf"))
        d = self.softmax(d)

        # recombine the outputs from a linear combination of weights and inputs
        x = torch.bmm(d.permute(0, 2, 1), x)

        return x
