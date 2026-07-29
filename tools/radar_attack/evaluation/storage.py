"""Persistence for raw adversarial 4D-radar point clouds."""

import json
import re
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch


def _safe_name(frame_id, fallback: int) -> str:
    if frame_id is None:
        return f'{fallback:06d}'
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(frame_id))
    return value or f'{fallback:06d}'


class AdversarialPointCloudWriter:
    """Save raw points and a JSONL manifest for reproducible evaluation."""

    def __init__(
        self,
        output_dir: Path,
        feature_names: Sequence[str],
        file_format: str = 'npy',
        run_metadata: Dict = None,
    ):
        if file_format not in {'npy', 'bin'}:
            raise ValueError('file_format must be "npy" or "bin"')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = list(feature_names)
        self.file_format = file_format
        self.run_metadata = dict(run_metadata or {})
        self.manifest_path = self.output_dir / 'manifest.jsonl'
        self.manifest_path.write_text('', encoding='utf-8')
        self.saved_samples = 0

    def save_batch(
        self,
        original_points: torch.Tensor,
        adversarial_points: torch.Tensor,
        batch_dict: Dict,
    ) -> None:
        if original_points.shape != adversarial_points.shape:
            raise ValueError('clean and adversarial point tensors must have equal shape')

        batch_indices = adversarial_points[:, 0].long()
        frame_ids = batch_dict.get('frame_id')
        batch_size = int(batch_dict['batch_size'])

        for batch_index in range(batch_size):
            point_mask = batch_indices == batch_index
            clean = original_points[point_mask, 1:].detach().cpu().numpy()
            adversarial = (
                adversarial_points[point_mask, 1:]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            frame_id = (
                frame_ids[batch_index] if frame_ids is not None else None
            )
            stem = _safe_name(frame_id, self.saved_samples)
            path = self.output_dir / f'{stem}.{self.file_format}'

            if self.file_format == 'npy':
                np.save(path, adversarial)
            else:
                adversarial.tofile(path)

            delta = np.abs(adversarial - clean)
            record = {
                'frame_id': str(frame_id) if frame_id is not None else stem,
                'file': path.name,
                'format': self.file_format,
                'shape': list(adversarial.shape),
                'dtype': str(adversarial.dtype),
                'feature_names': self.feature_names,
                'max_abs_delta': float(delta.max()) if delta.size else 0.0,
                'mean_abs_delta': float(delta.mean()) if delta.size else 0.0,
                **self.run_metadata,
            }
            with self.manifest_path.open('a', encoding='utf-8') as manifest:
                manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.saved_samples += 1
