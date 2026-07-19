import gc
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from backend.app.config import CLASS_NAMES, MODEL_CONFIGS
from backend.app.services.model_service import model_service
from cam.cam_factory import create_cam_method, get_target_layers


QUICK_CAM_METHODS = [
    "gradcam",
    "layercam",
    "eigencam",
]

FULL_CAM_METHODS = [
    "gradcam",
    "gradcamplusplus",
    "layercam",
    "eigencam",
    "scorecam",
    "hirescam",
    "xgradcam",
    "ablationcam",
]

CAM_DISPLAY_NAMES = {
    "gradcam": "Grad-CAM",
    "gradcamplusplus": "Grad-CAM++",
    "layercam": "LayerCAM",
    "eigencam": "EigenCAM",
    "scorecam": "ScoreCAM",
    "hirescam": "HiResCAM",
    "xgradcam": "XGrad-CAM",
    "ablationcam": "Ablation-CAM",
}

EDITED_CLASS_INDEX = 0
REAL_CLASS_INDEX = 1

cam_generation_lock = threading.Lock()


def normalize_heatmap(
    grayscale_cam: np.ndarray,
) -> np.ndarray:
    grayscale_cam = np.nan_to_num(
        grayscale_cam,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return np.clip(
        grayscale_cam,
        0.0,
        1.0,
    )


def save_rgb_image(
    rgb_array: np.ndarray,
    save_path: Path,
) -> None:
    image_uint8 = (
        np.clip(rgb_array, 0.0, 1.0) * 255
    ).astype(np.uint8)

    image_bgr = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        str(save_path),
        image_bgr,
    )


def release_cam(cam_object) -> None:
    try:
        cam_object.activations_and_grads.release()
    except Exception:
        pass

    del cam_object
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_model_names(
    model_names: Optional[List[str]],
) -> List[str]:
    if not model_names:
        return list(MODEL_CONFIGS.keys())

    invalid_models = [
        model_name
        for model_name in model_names
        if model_name not in MODEL_CONFIGS
    ]

    if invalid_models:
        raise ValueError(
            f"Unsupported models: {invalid_models}"
        )

    return model_names


def validate_cam_methods(
    methods: Optional[List[str]],
) -> List[str]:
    if not methods:
        return QUICK_CAM_METHODS

    invalid_methods = [
        method
        for method in methods
        if method not in FULL_CAM_METHODS
    ]

    if invalid_methods:
        raise ValueError(
            f"Unsupported CAM methods: {invalid_methods}"
        )

    return methods


def get_cam_target_information(
    prediction_index: int,
    target_mode: str,
) -> Dict:
    if target_mode not in {
        "predicted",
        "edited",
    }:
        raise ValueError(
            "target_mode must be 'predicted' or 'edited'."
        )

    if target_mode == "edited":
        target_index = EDITED_CLASS_INDEX

        return {
            "target_index": target_index,
            "target_class": "edited",
            "interpretation": "edited_class_evidence",
            "manipulation_localization_available": (
                prediction_index == EDITED_CLASS_INDEX
            ),
            "warning": (
                "This heatmap shows evidence associated with "
                "the edited class. It is not a confirmed "
                "manipulation mask."
            ),
        }

    target_index = prediction_index
    target_class = CLASS_NAMES[target_index]

    if prediction_index == EDITED_CLASS_INDEX:
        return {
            "target_index": target_index,
            "target_class": target_class,
            "interpretation": "suspected_edit_evidence",
            "manipulation_localization_available": True,
            "warning": (
                "This heatmap highlights regions that influenced "
                "the edited-class prediction. It is explanatory "
                "evidence, not an exact segmentation mask."
            ),
        }

    return {
        "target_index": target_index,
        "target_class": target_class,
        "interpretation": "authentic_class_explanation",
        "manipulation_localization_available": False,
        "warning": (
            "This heatmap explains the authentic-class prediction. "
            "It must not be interpreted as a manipulated region."
        ),
    }


def generate_cam_results(
    image_tensor: torch.Tensor,
    rgb_array: np.ndarray,
    static_directory: Path,
    methods: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    target_mode: str = "predicted",
) -> Dict:
    if not model_service.loaded:
        raise RuntimeError(
            "Models have not been loaded."
        )

    selected_methods = validate_cam_methods(
        methods
    )

    selected_models = validate_model_names(
        model_names
    )

    analysis_id = uuid.uuid4().hex

    result_root = (
        static_directory
        / "results"
        / analysis_id
    )

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_image_path = (
        result_root / "processed_input.png"
    )

    save_rgb_image(
        rgb_array,
        original_image_path,
    )

    input_tensor = image_tensor.to(
        model_service.device
    )

    model_results = []
    total_start = time.perf_counter()

    with cam_generation_lock:
        for model_name in selected_models:
            model_info = MODEL_CONFIGS[
                model_name
            ]

            model = model_service.get_model(
                model_name
            )

            model_directory = (
                result_root / model_name
            )

            model_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            classification_start = (
                time.perf_counter()
            )

            with torch.inference_mode():
                logits = model(input_tensor)

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )[0]

                predicted_index = int(
                    torch.argmax(
                        probabilities
                    ).item()
                )

            classification_runtime = (
                time.perf_counter()
                - classification_start
            )

            prediction = CLASS_NAMES[
                predicted_index
            ]

            confidence = float(
                probabilities[
                    predicted_index
                ].item()
            )

            probability_edited = float(
                probabilities[0].item()
            )

            probability_real = float(
                probabilities[1].item()
            )

            target_information = (
                get_cam_target_information(
                    prediction_index=predicted_index,
                    target_mode=target_mode,
                )
            )

            cam_target_index = (
                target_information[
                    "target_index"
                ]
            )

            target_layers, reshape_transform = (
                get_target_layers(
                    model,
                    model_name,
                )
            )

            cam_results = []

            for method_name in selected_methods:
                heatmap_path = (
                    model_directory
                    / f"{method_name}_heatmap.png"
                )

                overlay_path = (
                    model_directory
                    / f"{method_name}_overlay.png"
                )

                cam_object = None
                cam_start = time.perf_counter()

                try:
                    cam_object = create_cam_method(
                        method_name=method_name,
                        model=model,
                        model_name=model_name,
                        target_layers=target_layers,
                        reshape_transform=reshape_transform,
                    )

                    targets = [
                        ClassifierOutputTarget(
                            cam_target_index
                        )
                    ]

                    grayscale_cam = cam_object(
                        input_tensor=input_tensor,
                        targets=targets,
                    )[0]

                    cam_runtime = (
                        time.perf_counter()
                        - cam_start
                    )

                    grayscale_cam = (
                        normalize_heatmap(
                            grayscale_cam
                        )
                    )

                    overlay = show_cam_on_image(
                        rgb_array,
                        grayscale_cam,
                        use_rgb=True,
                    )

                    heatmap_uint8 = (
                        grayscale_cam * 255
                    ).astype(np.uint8)

                    cv2.imwrite(
                        str(heatmap_path),
                        heatmap_uint8,
                    )

                    cv2.imwrite(
                        str(overlay_path),
                        cv2.cvtColor(
                            overlay,
                            cv2.COLOR_RGB2BGR,
                        ),
                    )

                    status = "success"
                    error_message = ""

                    heatmap_url = (
                        f"/static/results/"
                        f"{analysis_id}/"
                        f"{model_name}/"
                        f"{method_name}_heatmap.png"
                    )

                    overlay_url = (
                        f"/static/results/"
                        f"{analysis_id}/"
                        f"{model_name}/"
                        f"{method_name}_overlay.png"
                    )

                except Exception as error:
                    cam_runtime = (
                        time.perf_counter()
                        - cam_start
                    )

                    status = "error"
                    error_message = str(error)
                    heatmap_url = None
                    overlay_url = None

                finally:
                    if cam_object is not None:
                        release_cam(
                            cam_object
                        )

                cam_results.append(
                    {
                        "method": method_name,
                        "display_name": (
                            CAM_DISPLAY_NAMES[
                                method_name
                            ]
                        ),
                        "status": status,
                        "error_message": (
                            error_message
                        ),
                        "runtime_seconds": (
                            cam_runtime
                        ),
                        "heatmap_url": (
                            heatmap_url
                        ),
                        "overlay_url": (
                            overlay_url
                        ),
                        "target_class": (
                            target_information[
                                "target_class"
                            ]
                        ),
                        "interpretation": (
                            target_information[
                                "interpretation"
                            ]
                        ),
                        "warning": (
                            target_information[
                                "warning"
                            ]
                        ),
                    }
                )

            model_results.append(
                {
                    "model_name": model_name,
                    "display_name": (
                        model_info[
                            "display_name"
                        ]
                    ),
                    "prediction": prediction,
                    "confidence": confidence,
                    "probability_edited": (
                        probability_edited
                    ),
                    "probability_real": (
                        probability_real
                    ),
                    "classification_runtime_seconds": (
                        classification_runtime
                    ),
                    "reference_test_accuracy": (
                        model_info[
                            "reference_accuracy"
                        ]
                    ),
                    "reference_test_f1": (
                        model_info[
                            "reference_f1"
                        ]
                    ),
                    "cam_target_class": (
                        target_information[
                            "target_class"
                        ]
                    ),
                    "cam_interpretation": (
                        target_information[
                            "interpretation"
                        ]
                    ),
                    "manipulation_localization_available": (
                        target_information[
                            "manipulation_localization_available"
                        ]
                    ),
                    "cam_warning": (
                        target_information[
                            "warning"
                        ]
                    ),
                    "cam_results": cam_results,
                }
            )

    total_runtime = (
        time.perf_counter() - total_start
    )

    edited_votes = sum(
        result["prediction"] == "edited"
        for result in model_results
    )

    real_votes = (
        len(model_results) - edited_votes
    )

    if edited_votes > real_votes:
        consensus_prediction = "edited"
        agreement_votes = edited_votes

        consensus_warning = (
            "Most models predicted that the image is edited. "
            "The heatmaps show model evidence associated with "
            "the prediction, not confirmed manipulation masks."
        )

        default_show_heatmaps = True

    else:
        consensus_prediction = "real"
        agreement_votes = real_votes

        consensus_warning = (
            "Most models predicted that the image is authentic. "
            "Heatmaps are hidden by default because attention "
            "maps on authentic images must not be interpreted "
            "as manipulated regions."
        )

        default_show_heatmaps = False

    total_models = len(model_results)

    return {
        "analysis_id": analysis_id,
        "processed_image_url": (
            f"/static/results/"
            f"{analysis_id}/"
            f"processed_input.png"
        ),
        "methods": selected_methods,
        "selected_models": selected_models,
        "target_mode": target_mode,
        "total_runtime_seconds": (
            total_runtime
        ),
        "consensus": {
            "prediction": (
                consensus_prediction
            ),
            "edited_votes": edited_votes,
            "real_votes": real_votes,
            "agreement_votes": (
                agreement_votes
            ),
            "total_models": total_models,
            "agreement_percentage": (
                agreement_votes
                / total_models
            ),
            "average_probability_edited": (
                sum(
                    result[
                        "probability_edited"
                    ]
                    for result in model_results
                )
                / total_models
            ),
            "average_probability_real": (
                sum(
                    result[
                        "probability_real"
                    ]
                    for result in model_results
                )
                / total_models
            ),
            "manipulation_detected": (
                consensus_prediction
                == "edited"
            ),
            "default_show_heatmaps": (
                default_show_heatmaps
            ),
            "warning": (
                consensus_warning
            ),
        },
        "models": model_results,
    }