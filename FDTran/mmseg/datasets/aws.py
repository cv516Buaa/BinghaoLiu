# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class AWS16KDataset(BaseSegDataset):
    """Africa Water Segmentation 16K dataset.

    In segmentation map annotation for AWS16K dataset, 0 is the background index.
    ``reduce_zero_label`` should be set to False.
    """
    METAINFO = dict(
        classes=('background', 'water'),
        palette=[[0, 0, 0], [240, 240, 240]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
