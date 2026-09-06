import torch
import torch.nn as nn


class PillarAttention(nn.Module):
    """Self-attention across the occupied pillars of each batch item."""

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.attn_channels = model_cfg.get('ATTN_CHANNELS', input_channels)
        self.num_point_features = self.attn_channels
        self.ffn_hidden = model_cfg.get('FFN_CHANNELS', self.attn_channels * 2)

        self.pre_mlp = (
            nn.Linear(input_channels, self.attn_channels)
            if input_channels != self.attn_channels else nn.Identity()
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_channels,
            num_heads=model_cfg.NUM_HEADS,
            dropout=model_cfg.get('DROPOUT', 0.0),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.attn_channels)
        self.ffn = nn.Sequential(
            nn.Linear(self.attn_channels, self.ffn_hidden),
            nn.GELU(),
            nn.Linear(self.ffn_hidden, self.attn_channels),
        )
        self.norm2 = nn.LayerNorm(self.attn_channels)

    def forward(self, batch_dict):
        pillar_features = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']
        if pillar_features.shape[0] == 0:
            return batch_dict

        batch_size = int(batch_dict.get('batch_size', coords[:, 0].max().item() + 1))
        pillar_counts = [(coords[:, 0] == batch_idx).sum().item() for batch_idx in range(batch_size)]
        nonempty_counts = [count for count in pillar_counts if count]
        if not nonempty_counts:
            return batch_dict
        max_pillars = max(nonempty_counts)

        padded_features = pillar_features.new_zeros(
            (batch_size, max_pillars, pillar_features.shape[-1])
        )
        key_padding_mask = torch.ones(
            (batch_size, max_pillars), dtype=torch.bool, device=pillar_features.device
        )
        for batch_idx, count in enumerate(pillar_counts):
            if count == 0:
                continue
            batch_mask = coords[:, 0] == batch_idx
            padded_features[batch_idx, :count] = pillar_features[batch_mask]
            key_padding_mask[batch_idx, :count] = False

        # MultiheadAttention produces NaNs when a row is entirely masked. Give
        # empty batch rows one harmless zero token, then discard it below.
        empty_rows = key_padding_mask.all(dim=1)
        key_padding_mask[empty_rows, 0] = False

        x = self.pre_mlp(padded_features)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        batch_dict['pillar_features'] = torch.cat([
            x[batch_idx, :count]
            for batch_idx, count in enumerate(pillar_counts)
            if count
        ], dim=0)
        return batch_dict
