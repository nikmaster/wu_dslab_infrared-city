#This notebook is based on the code of:
#V. S. F. Garnot and L. Landrieu, “Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 4872–4881, doi: 10.1109/ICCV48922.2021.00483.

import copy

import numpy as np
import torch
import torch.nn as nn


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        # Lightweight Temporal Attention Encoder for image time series

        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        # Optional channel projection
        if d_model is not None:
            self.d_model = d_model

            # 1D convolution projects features into attention dimension
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        # Ensure input dimension matches model dimension
        assert self.mlp[0] == self.d_model

        # Positional encoding
        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        # Multi-head temporal attention
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)

        # Input and output normalization
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels,)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1],)

        # MLP refinement layers
        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        # Forward pass

        sz_b, seq_len, d, h, w = x.shape

        # Expand padding mask spatially
        if pad_mask is not None:
            pad_mask = (pad_mask.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w)))  # BxTxHxW

            # Reshape into pixel-wise temporal sequences
            pad_mask = (pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len))

        # Rearrange image sequence into pixel sequences
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)

        # Normalize input channels
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        # Project features into model dimension
        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        # Add positional encoding
        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        # Multi-head temporal attention
        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        # Concatenate attention heads
        out = (out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1))

        # Output dropout, normalization and reshaping
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        # Reshape attention maps
        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    # Multi-Head Attention module
    # Modified from github.com/jadore801120/attention-is-all-you-need-pytorch

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        # Learnable query vectors
        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        # Key projection layer
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        # Forward pass

        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        # Query tensor
        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        # Generate keys
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        # Replicate padding mask for each head
        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))

        # Split value tensor across heads
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)

        # Attention computation
        if return_comp:
            output, attn, comp = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
        else:
            output, attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)

        # Reshape attention weights
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    #Scaled Dot-Product Attention
    #Modified from github.com/jadore801120/attention-is-all-you-need-pytorch

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        # Forward pass

        # Compute compatibility scores and scale
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        # Mask padded timestamps
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        
        # Store compatibility scores if requested
        if return_comp:
            comp = attn

        # Normalize attention scores
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # Weighted value aggregation
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn

class PositionalEncoder(nn.Module):
    # Positional Encoder

    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat

        # Precompute denominators
        self.denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self.updated_location = False

    def forward(self, batch_positions):
        # Forward pass

        # Move denominators to correct device
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True

        # Compute sinusoidal embeddings
        sinusoid_table = (batch_positions[:, :, None] / self.denom[None, None, :])  # B x T x C

        # Apply sine to even indices and cosine to odd indices
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        # Repeat encoding for all heads
        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table