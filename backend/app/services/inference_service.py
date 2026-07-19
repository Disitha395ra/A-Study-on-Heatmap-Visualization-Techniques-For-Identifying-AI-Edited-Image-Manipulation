import time

import torch

from backend.app.config import (
    CLASS_NAMES,
    MODEL_CONFIGS,
)
from backend.app.services.model_service import model_service


def classify_with_all_models(
    image_tensor: torch.Tensor,
) -> dict:
    if not model_service.loaded:
        raise RuntimeError(
            "Models are not loaded."
        )

    image_tensor = image_tensor.to(
        model_service.device
    )

    model_results = []

    edited_votes = 0
    real_votes = 0

    edited_probability_sum = 0.0
    real_probability_sum = 0.0

    for model_name, model_info in MODEL_CONFIGS.items():
        model = model_service.get_model(model_name)

        inference_start = time.perf_counter()

        with torch.inference_mode():
            logits = model(image_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

            predicted_index = int(
                torch.argmax(probabilities).item()
            )

        inference_time = (
            time.perf_counter() - inference_start
        )

        edited_probability = float(
            probabilities[0].item()
        )

        real_probability = float(
            probabilities[1].item()
        )

        prediction = CLASS_NAMES[predicted_index]

        confidence = float(
            probabilities[predicted_index].item()
        )

        if prediction == "edited":
            edited_votes += 1
        else:
            real_votes += 1

        edited_probability_sum += edited_probability
        real_probability_sum += real_probability

        model_results.append(
            {
                "model_name": model_name,
                "display_name": model_info["display_name"],
                "prediction": prediction,
                "confidence": confidence,
                "probability_edited": edited_probability,
                "probability_real": real_probability,
                "inference_time_seconds": inference_time,

                # Historical test-set values, not current image accuracy.
                "reference_test_accuracy": (
                    model_info["reference_accuracy"]
                ),
                "reference_test_f1": (
                    model_info["reference_f1"]
                ),
            }
        )

    total_models = len(model_results)

    if edited_votes > real_votes:
        consensus_prediction = "edited"
        agreement_votes = edited_votes

    elif real_votes > edited_votes:
        consensus_prediction = "real"
        agreement_votes = real_votes

    else:
        # This branch is unlikely with 5 models, but it is kept safe.
        average_edited_probability = (
            edited_probability_sum / total_models
        )

        consensus_prediction = (
            "edited"
            if average_edited_probability >= 0.5
            else "real"
        )

        agreement_votes = max(
            edited_votes,
            real_votes,
        )

    average_edited_probability = (
        edited_probability_sum / total_models
    )

    average_real_probability = (
        real_probability_sum / total_models
    )

    return {
        "consensus": {
            "prediction": consensus_prediction,
            "edited_votes": edited_votes,
            "real_votes": real_votes,
            "agreement_votes": agreement_votes,
            "total_models": total_models,
            "agreement_percentage": (
                agreement_votes / total_models
            ),
            "average_probability_edited": (
                average_edited_probability
            ),
            "average_probability_real": (
                average_real_probability
            ),
        },
        "models": model_results,
    }