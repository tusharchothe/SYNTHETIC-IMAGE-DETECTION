from .inference_service import (
    InferenceResult,
    run_clip_prediction,
    run_resnet_prediction,
    run_prediction_for_model,
)

__all__ = [
    "InferenceResult",
    "run_resnet_prediction",
    "run_clip_prediction",
    "run_prediction_for_model",
]
