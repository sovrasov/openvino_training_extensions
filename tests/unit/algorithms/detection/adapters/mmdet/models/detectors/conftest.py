# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
import uuid
from typing import Dict

import datumaro as dm
import mmcv
import numpy as np
import pytest
from mmdet.datasets import build_dataloader as mmdet_build_dataloader
from mmdet.datasets import build_dataset as mmdet_build_dataset

from otx.algorithms.common.adapters.mmcv.utils.builder import (
    build_dataloader,
    build_dataset,
)
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model_template import TaskType
from otx.core.data.adapter.detection_dataset_adapter import DetectionDatasetAdapter


@pytest.fixture
def fxt_cfg_custom_atss(num_classes: int = 2) -> Dict:
    train_cfg = mmcv.Config(
        dict(
            assigner=dict(type="ATSSAssigner", topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        )
    )
    cfg = dict(
        type="CustomATSS",
        backbone=dict(
            avg_down=False,
            base_channels=64,
            conv_cfg=None,
            dcn=None,
            deep_stem=False,
            depth=18,
            dilations=(1, 1, 1, 1),
            frozen_stages=-1,
            in_channels=3,
            init_cfg=None,
            norm_cfg=dict(requires_grad=True, type="BN"),
            norm_eval=True,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            plugins=None,
            pretrained=None,
            stage_with_dcn=(False, False, False, False),
            stem_channels=None,
            strides=(1, 2, 2, 2),
            style="pytorch",
            type="mmdet.ResNet",
            with_cp=False,
            zero_init_residual=True,
        ),
        neck=dict(
            type="FPN",
            in_channels=[64, 128, 256, 512],
            out_channels=64,
            start_level=1,
            add_extra_convs="on_output",
            num_outs=5,
            relu_before_extra_convs=True,
        ),
        bbox_head=dict(
            type="CustomATSSHead",
            num_classes=num_classes,
            in_channels=64,
            stacked_convs=4,
            feat_channels=64,
            anchor_generator=dict(
                type="AnchorGenerator",
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            use_qfl=False,
            qfl_cfg=dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0),
        ),
        train_cfg=train_cfg,
    )
    return cfg


@pytest.fixture
def fxt_cfg_custom_ssd(num_classes: int = 2) -> Dict:
    train_cfg = mmcv.Config(
        {
            "assigner": {
                "type": "MaxIoUAssigner",
                "min_pos_iou": 0.0,
                "ignore_iof_thr": -1,
                "gt_max_assign_all": False,
                "pos_iou_thr": 0.4,
                "neg_iou_thr": 0.4,
            },
            "smoothl1_beta": 1.0,
            "allowed_border": -1,
            "pos_weight": -1,
            "neg_pos_ratio": 3,
            "debug": False,
            "use_giou": False,
            "use_focal": False,
        }
    )

    cfg = dict(
        type="CustomSingleStageDetector",
        backbone=dict(
            type="mmdet.mobilenetv2_w1", out_indices=(4, 5), frozen_stages=-1, norm_eval=False, pretrained=True
        ),
        neck=None,
        bbox_head=dict(
            type="CustomSSDHead",
            num_classes=num_classes,
            in_channels=(int(96.0), int(320.0)),
            use_depthwise=True,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU"),
            init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
            loss_balancing=False,
            anchor_generator=dict(
                type="SSDAnchorGeneratorClustered",
                strides=(16, 32),
                reclustering_anchors=True,
                widths=[
                    [
                        38.641007923271076,
                        92.49516032784699,
                        271.4234764938237,
                        141.53469410876247,
                    ],
                    [
                        206.04136086566515,
                        386.6542727907841,
                        716.9892752215089,
                        453.75609561761405,
                        788.4629155558277,
                    ],
                ],
                heights=[
                    [
                        48.9243877087132,
                        147.73088476194903,
                        158.23569788707474,
                        324.14510379107367,
                    ],
                    [
                        587.6216059488938,
                        381.60024152086544,
                        323.5988913027747,
                        702.7486097568518,
                        741.4865860938451,
                    ],
                ],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=(0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2),
            ),
        ),
        train_cfg=train_cfg,
    )

    return cfg