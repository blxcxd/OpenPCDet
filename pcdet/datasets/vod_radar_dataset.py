from .kitti.kitti_dataset import KittiDataset


class VoDRadarDataset(KittiDataset):
    """VoD radar data stored in the repository's KITTI-compatible layout.

    Prediction serialization is inherited from ``KittiDataset``. Official VoD
    evaluation is intentionally kept in ``tools/eval_vod.py`` because the KITTI
    evaluator uses different metrics and is not valid for this dataset.
    """

    def evaluation(self, det_annos, class_names, **kwargs):
        message = (
            'VoD predictions were generated successfully. '
            'Run tools/eval_vod.py on result.pkl for official VoD metrics.'
        )
        return message, {}
