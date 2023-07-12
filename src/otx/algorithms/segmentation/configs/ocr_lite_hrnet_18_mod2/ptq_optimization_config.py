"""PTQ config file."""
from nncf import IgnoredScope
from nncf.common.quantization.structs import QuantizationPreset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.range_estimator import (
    AggregatorType,
    RangeEstimatorParameters,
    StatisticsCollectorParameters,
    StatisticsType,
)

advanced_parameters = AdvancedQuantizationParameters(
    activations_range_estimator_params=RangeEstimatorParameters(
        min=StatisticsCollectorParameters(
            statistics_type=StatisticsType.QUANTILE, aggregator_type=AggregatorType.MIN, quantile_outlier_prob=1e-4
        ),
        max=StatisticsCollectorParameters(
            statistics_type=StatisticsType.QUANTILE, aggregator_type=AggregatorType.MAX, quantile_outlier_prob=1e-4
        ),
    ),
    backend_params={"use_pot": True},
)

preset = QuantizationPreset.MIXED

ignored_scope = IgnoredScope(
    names=[
        "/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul",
        "/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul_1",
        "/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul",
        "/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul_1",
        "/backbone/stage0/stage0.0/Add_1",
        "/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul",
        "/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul_1",
        "/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul",
        "/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul_1",
        "/backbone/stage0/stage0.1/Add_1",
        "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul",
        "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_1",
        "/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_2",
        "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul",
        "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_1",
        "/backbone/stage1/stage1.0/Add_1",
        "/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_2",
        "/backbone/stage1/stage1.0/Add_2",
        "/backbone/stage1/stage1.0/Add_5",
        "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul",
        "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_1",
        "/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_2",
        "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul",
        "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_1",
        "/backbone/stage1/stage1.1/Add_1",
        "/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_2",
        "/backbone/stage1/stage1.1/Add_2",
        "/backbone/stage1/stage1.1/Add_5",
        "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul",
        "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_1",
        "/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_2",
        "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul",
        "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_1",
        "/backbone/stage1/stage1.2/Add_1",
        "/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_2",
        "/backbone/stage1/stage1.2/Add_2",
        "/backbone/stage1/stage1.2/Add_5",
        "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul",
        "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_1",
        "/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_2",
        "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul",
        "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_1",
        "/backbone/stage1/stage1.3/Add_1",
        "/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_2",
        "/backbone/stage1/stage1.3/Add_2",
        "/backbone/stage1/stage1.3/Add_5",
        "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul",
        "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_1",
        "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_2",
        "/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_3",
        "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul",
        "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_1",
        "/backbone/stage2/stage2.0/Add_1",
        "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_2",
        "/backbone/stage2/stage2.0/Add_2",
        "/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_3",
        "/backbone/stage2/stage2.0/Add_3",
        "/backbone/stage2/stage2.0/Add_6",
        "/backbone/stage2/stage2.0/Add_7",
        "/backbone/stage2/stage2.0/Add_11",
        "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul",
        "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_1",
        "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_2",
        "/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_3",
        "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul",
        "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_1",
        "/backbone/stage2/stage2.1/Add_1",
        "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_2",
        "/backbone/stage2/stage2.1/Add_2",
        "/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_3",
        "/backbone/stage2/stage2.1/Add_3",
        "/backbone/stage2/stage2.1/Add_6",
        "/backbone/stage2/stage2.1/Add_7",
        "/backbone/stage2/stage2.1/Add_11",
        "/aggregator/Add",
        "/aggregator/Add_1",
        "/aggregator/Add_2",
        "/backbone/stage2/stage2.1/Add",
    ]
)