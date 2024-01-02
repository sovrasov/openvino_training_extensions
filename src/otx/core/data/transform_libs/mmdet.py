# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMDET data transform functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from datumaro import Polygon
from mmdet.datasets.transforms import (
    LoadAnnotations as MMDetLoadAnnotations,
)
from mmdet.datasets.transforms import (
    PackDetInputs as MMDetPackDetInputs,
)
from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.entity.instance_segmentation import InstanceSegDataEntity

from .mmcv import MMCVTransformLib

if TYPE_CHECKING:
    from mmdet.structures.det_data_sample import DetDataSample
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


@TRANSFORMS.register_module(force=True)
class LoadAnnotations(MMDetLoadAnnotations):
    """Class to override MMDet LoadAnnotations."""

    def transform(self, results: dict) -> dict:
        """Transform OTXDataEntity to MMDet annotation data entity format."""
        if (otx_data_entity := results.get("__otx__")) is None:
            msg = "__otx__ key should be passed from the previous pipeline (LoadImageFromFile)"
            raise RuntimeError(msg)

        if self.with_bbox and isinstance(otx_data_entity, (DetDataEntity, InstanceSegDataEntity)):
            gt_bboxes = otx_data_entity.bboxes.numpy()
            results["gt_bboxes"] = gt_bboxes
        if self.with_label and isinstance(otx_data_entity, (DetDataEntity, InstanceSegDataEntity)):
            gt_bboxes_labels = otx_data_entity.labels.numpy()
            results["gt_bboxes_labels"] = gt_bboxes_labels
            results["gt_ignore_flags"] = np.zeros_like(gt_bboxes_labels, dtype=np.bool_)
        if self.with_mask and isinstance(otx_data_entity, InstanceSegDataEntity):
            height, width = results["ori_shape"]
            gt_masks = self._generate_gt_masks(otx_data_entity, height, width)
            results["gt_masks"] = gt_masks
        return results

    def _generate_gt_masks(
        self,
        otx_data_entity: InstanceSegDataEntity,
        height: int,
        width: int,
    ) -> BitmapMasks | PolygonMasks:
        """Generate ground truth masks based on the given otx_data_entity.

        Args:
            otx_data_entity (OTXDataEntity): The data entity containing the masks or polygons.
            height (int): The height of the masks.
            width (int): The width of the masks.

        Returns:
            gt_masks (BitmapMasks or PolygonMasks): The generated ground truth masks.
        """
        if len(otx_data_entity.masks):
            gt_masks = BitmapMasks(otx_data_entity.masks.numpy(), height, width)
        else:
            gt_masks = PolygonMasks(
                [[np.array(polygon.points)] for polygon in otx_data_entity.polygons],
                height,
                width,
            )
        return gt_masks


@TRANSFORMS.register_module(force=True)
class PackDetInputs(MMDetPackDetInputs):
    """Class to override PackDetInputs LoadAnnotations."""

    def transform(self, results: dict) -> DetDataEntity | InstanceSegDataEntity:
        """Pack MMDet data entity into DetDataEntity or InstanceSegDataEntity."""
        otx_data_entity = results["__otx__"]

        if isinstance(otx_data_entity, DetDataEntity):
            return self.pack_det_inputs(results)
        if isinstance(otx_data_entity, InstanceSegDataEntity):
            return self.pack_inst_inputs(results)
        msg = "Unsupported data entity type"
        raise TypeError(msg)

    def pack_det_inputs(self, results: dict) -> DetDataEntity:
        """Pack MMDet data entity into DetDataEntity."""
        transformed = super().transform(results)
        data_samples = transformed["data_samples"]
        img_shape, ori_shape, pad_shape, scale_factor = self.extract_metadata(data_samples)

        bboxes = self.convert_bboxes(data_samples.gt_instances.bboxes, img_shape)
        labels = data_samples.gt_instances.labels

        return DetDataEntity(
            image=tv_tensors.Image(transformed.get("inputs")),
            img_info=self.create_image_info(0, img_shape, ori_shape, pad_shape, scale_factor),
            bboxes=bboxes,
            labels=labels,
        )

    def pack_inst_inputs(self, results: dict) -> InstanceSegDataEntity:
        """Pack MMDet data entity into InstanceSegDataEntity."""
        transformed = super().transform(results)
        data_samples = transformed["data_samples"]
        img_shape, ori_shape, pad_shape, scale_factor = self.extract_metadata(data_samples)

        bboxes = self.convert_bboxes(data_samples.gt_instances.bboxes, img_shape)
        labels = data_samples.gt_instances.labels
        image_info = self.create_image_info(0, img_shape, ori_shape, pad_shape, scale_factor)

        masks, polygons = self.convert_masks_and_polygons(data_samples.gt_instances.masks)

        return InstanceSegDataEntity(
            image=tv_tensors.Image(transformed.get("inputs")),
            img_info=image_info,
            bboxes=bboxes,
            masks=masks,
            labels=labels,
            polygons=polygons,
        )

    def extract_metadata(
        self,
        data_samples: DetDataSample,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[float, float]]:
        """Extract metadata from data_samples."""
        img_shape = data_samples.img_shape
        ori_shape = data_samples.ori_shape
        pad_shape = data_samples.metainfo.get("pad_shape", img_shape)
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))
        return img_shape, ori_shape, pad_shape, scale_factor

    def convert_bboxes(self, original_bboxes: torch.Tensor, img_shape: tuple[int, int]) -> tv_tensors.BoundingBoxes:
        """Convert bounding boxes to tv_tensors.BoundingBoxes format."""
        return tv_tensors.BoundingBoxes(
            original_bboxes.float(),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=img_shape,
        )

    def create_image_info(
        self,
        img_idx: int,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        pad_shape: tuple[int, int],
        scale_factor: tuple[float, float],
    ) -> ImageInfo:
        """Create ImageInfo instance."""
        return ImageInfo(
            img_idx=img_idx,
            img_shape=img_shape,
            ori_shape=ori_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
        )

    def convert_masks_and_polygons(self, masks: BitmapMasks | PolygonMasks) -> tuple[tv_tensors.Mask, list[Polygon]]:
        """Convert masks and polygons to the desired format."""
        if isinstance(masks, BitmapMasks):
            masks_tensor = tv_tensors.Mask(masks.to_ndarray(), dtype=torch.int8)
        else:
            masks_tensor = tv_tensors.Mask(torch.empty(0))

        polygons = [Polygon(polygon[0]) for polygon in masks.masks] if isinstance(masks, PolygonMasks) else []

        return masks_tensor, polygons


class MMDetTransformLib(MMCVTransformLib):
    """Helper to support MMDET transforms in OTX."""

    @classmethod
    def get_builder(cls) -> Registry:
        """Transform builder obtained from MMDet."""
        return TRANSFORMS

    @classmethod
    def generate(cls, config: SubsetConfig) -> list[Callable]:
        """Generate MMDET transforms from the configuration."""
        transforms = super().generate(config)

        cls._check_mandatory_transforms(
            transforms,
            mandatory_transforms={LoadAnnotations, PackDetInputs},
        )

        return transforms