#This notebook is based on the code of:
#V. S. F. Garnot and L. Landrieu, “Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 4872–4881, doi: 10.1109/ICCV48922.2021.00483.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltaeT import LTAE2d


class UTAE(nn.Module):
    # Main architecture with encoder, temporal attention and decoder

    def __init__(self,config):
        super(UTAE, self).__init__()

        # Get number of encoder/decoder stages
        self.n_stages = len(config.encoder_widths)

        # Channel sizes for encoder/decoder
        self.encoder_widths = config.encoder_widths
        self.decoder_widths = config.decoder_widths

        # Dimension of encoder input
        self.enc_dim = (config.decoder_widths[0] if config.decoder_widths is not None else config.encoder_widths[0])

        # Total stacked feature dim
        self.stack_dim = (sum(config.decoder_widths) if config.decoder_widths is not None else sum(config.encoder_widths))

        # Temporal padding value
        self.pad_value = config.pad_value

        # Validate that encoder and decoder are compatible
        if config.decoder_widths is not None:
            assert len(config.encoder_widths) == len(config.decoder_widths)
            assert config.encoder_widths[-1] == config.decoder_widths[-1]
        else:
            config.decoder_widths = config.encoder_widths

        # Initial convolution block
        self.in_conv = ConvBlock(
            nkernels=[config.input_dim] + [config.encoder_widths[0], config.encoder_widths[0]],
            pad_value=config.pad_value,
            norm=config.encoder_norm,
            padding_mode=config.padding_mode,
        )

        # Encoder
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=config.encoder_widths[i],
                d_out=config.encoder_widths[i + 1],
                k=config.str_conv_k,
                s=config.str_conv_s,
                p=config.str_conv_p,
                pad_value=config.pad_value,
                norm=config.encoder_norm,
                padding_mode=config.padding_mode,
            )
            for i in range(self.n_stages - 1)
        )

        # Decoder
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=config.decoder_widths[i],
                d_out=config.decoder_widths[i - 1],
                d_skip=config.encoder_widths[i - 1],
                k=config.str_conv_k,
                s=config.str_conv_s,
                p=config.str_conv_p,
                norm="batch",
                padding_mode=config.padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )

        # Temporal attention decoder, processes temporal sequence at bottleneck
        self.temporal_encoder = LTAE2d(
            in_channels=config.encoder_widths[-1],
            d_model=config.d_model,
            n_head=config.n_head,
            mlp=[config.d_model, config.encoder_widths[-1]],
            return_att=True,
            d_k=config.d_k,
        )

        # Temporal aggregation for skip connections
        self.temporal_aggregator = Temporal_Aggregator(mode=config.agg_mode)

        # Binary presence prediction head
        self.presence_head = ConvBlock(nkernels=[config.decoder_widths[0], 1],padding_mode=config.padding_mode,last_relu=False)

        # Count prediction head
        self.count_head = ConvBlock(nkernels=[config.decoder_widths[0], 1],padding_mode=config.padding_mode,last_relu=False)

    def forward(self, input, batch_positions=None, return_att=False):
        # Forward pass

        # Identify padded temporal frames
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )

        # Initial convolution and store encoder feature maps for skip connections
        out = self.in_conv.smart_forward(input)
        feature_maps = [out]

        # Spatial encoder
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # Temporal encoder
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )

        # Spatial decoder
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip)

        # Prediction heads
        presence_logits = self.presence_head(out)
        count_logits = self.count_head(out)

        if return_att:
            return presence_logits, count_logits, att
        else:
            return presence_logits, count_logits

class TemporallySharedBlock(nn.Module):
    # Help combine the batch and temporal dimension

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        # Can handle 4D as well as 5D tensors

        # If already 4D process normally
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            # Input shape
            b, t, c, h, w = input.shape

            # Compute output tensor
            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape
            
            # Merge batch and temporal dimensions
            out = input.view(b * t, c, h, w)

            if self.pad_value is not None:
                # Detect padded frames
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    # Create padded output tensor
                    temp = (torch.ones(self.out_shape, device=input.device, requires_grad=False) * self.pad_value)

                    # Process only frames which are not padded
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)

            # Restore temporal dimension
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    # Basic convolutional layer

    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []

        # Check which normalization should be used
        if norm == "batch":
                nl = lambda c: nn.GroupNorm(num_groups=min(8, c),num_channels=c)
        else:
            nl = None

        # Build convolutional stack
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )

            # Normalization
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            # Activation
            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    # Convolutional block

    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    # Downsampling convolutional block

    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)

        # Downsampling convolution
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )

        # Feature expansion
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

        # Residual refinement
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        # Forward pass

        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):
    # Upsampling convolutional block

    def __init__(self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"):
        super(UpConvBlock, self).__init__()

        # Skip feature dimension
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.GroupNorm(min(8, d), d),
            nn.ReLU(),
        )

        # Transposed convolution upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.GroupNorm(min(8, d_out), d_out),
            nn.ReLU(),
        )

        # Feature fusion convolutions
        self.conv1 = ConvLayer(nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, input, skip):
        # Forward pass

        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class Temporal_Aggregator(nn.Module):
    # Aggregates temporal feature maps

    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):

        if pad_mask is not None and pad_mask.any():
            # Handle padded sequences

            # Attention aggregation per head
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                # Resize attention to match feature map resolution
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])

                # Mask padded timestamps
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                # Split channels by attention head
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW

                # Weighted temporal aggregation
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW

                # Concatenate heads
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            
            elif self.mode == "att_mean":
                # Mean attention aggregation

                # Average attention over heads
                attn = attn_mask.mean(dim=0)  # -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)

                # Remove padded frames
                attn = attn * (~pad_mask).float()[:, :, None, None]

                # Weighted sum
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            
            elif self.mode == "mean":
                # Simple temporal mean

                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
            
        else:
            # Similar procedure but without padded frames

            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


