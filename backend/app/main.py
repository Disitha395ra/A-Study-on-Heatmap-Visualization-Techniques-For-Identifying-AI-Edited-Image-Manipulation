from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import (
    ALLOWED_CONTENT_TYPES,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    MODEL_CONFIGS,
    PROJECT_ROOT,
    get_checkpoint_path,
)
from backend.app.services.cam_service import (
    FULL_CAM_METHODS,
    QUICK_CAM_METHODS,
    generate_cam_results,
)

from backend.app.services.image_service import (
    ImageValidationError,
    process_uploaded_image,
)
from backend.app.services.inference_service import (
    classify_with_all_models,
)
from backend.app.services.model_service import (
    model_service,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "uploads").mkdir(
    parents=True,
    exist_ok=True,
)
(STATIC_DIR / "results").mkdir(
    parents=True,
    exist_ok=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_all_models()
    yield


app = FastAPI(
    title="AI-Edited Image Analysis API",
    description=(
        "Runs five trained classifiers and CAM-based "
        "visualization methods for AI-edited image analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get("/")
def root():
    return {
        "name": "AI-Edited Image Analysis API",
        "status": "running",
        "models": len(MODEL_CONFIGS),
        "cam_methods": 8,
    }


@app.get("/api/health")
def health_check():
    checkpoint_status = {
        model_name: get_checkpoint_path(
            model_name
        ).exists()
        for model_name in MODEL_CONFIGS
    }

    return {
        "status": "healthy",
        "project_root": str(PROJECT_ROOT),
        "model_service": model_service.model_status(),
        "checkpoint_status": checkpoint_status,
        "all_checkpoints_available": all(
            checkpoint_status.values()
        ),
    }


@app.get("/api/models")
def list_models():
    return {
        "models": [
            {
                "model_name": model_name,
                "display_name": model_info["display_name"],
                "reference_test_accuracy": (
                    model_info["reference_accuracy"]
                ),
                "reference_test_f1": (
                    model_info["reference_f1"]
                ),
            }
            for model_name, model_info in MODEL_CONFIGS.items()
        ]
    }


@app.post("/api/analyze/classify")
async def classify_uploaded_image(
    image: UploadFile = File(...),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image format. "
                "Upload a JPG, JPEG, PNG, or WebP image."
            ),
        )

    image_bytes = await image.read()

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image is too large. Maximum upload size "
                f"is {MAX_UPLOAD_SIZE_MB} MB."
            ),
        )

    try:
        processed_image = process_uploaded_image(
            image_bytes
        )

    except ImageValidationError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    classification_result = classify_with_all_models(
        processed_image.tensor
    )

    return {
        "filename": image.filename,
        "content_type": image.content_type,
        "image_information": {
            "original_width": (
                processed_image.original_width
            ),
            "original_height": (
                processed_image.original_height
            ),
            "original_format": (
                processed_image.original_format
            ),
            "processed_width": (
                processed_image.processed_width
            ),
            "processed_height": (
                processed_image.processed_height
            ),
            "preprocessing": (
                "EXIF correction, RGB conversion, "
                "direct resize to 224x224, "
                "ImageNet normalization"
            ),
        },
        **classification_result,
    }

@app.post("/api/analyze/quick")
async def quick_cam_analysis(
    image: UploadFile = File(...),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image format. "
                "Upload a JPG, JPEG, PNG, or WebP image."
            ),
        )

    image_bytes = await image.read()

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image is too large. Maximum upload size "
                f"is {MAX_UPLOAD_SIZE_MB} MB."
            ),
        )

    try:
        processed_image = process_uploaded_image(
            image_bytes
        )

        analysis_result = generate_cam_results(
            image_tensor=processed_image.tensor,
            rgb_array=processed_image.rgb_array,
            static_directory=STATIC_DIR,
            methods=QUICK_CAM_METHODS,
        )

    except ImageValidationError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "filename": image.filename,
        "content_type": image.content_type,
        "mode": "quick",
        "image_information": {
            "original_width": (
                processed_image.original_width
            ),
            "original_height": (
                processed_image.original_height
            ),
            "original_format": (
                processed_image.original_format
            ),
            "processed_width": (
                processed_image.processed_width
            ),
            "processed_height": (
                processed_image.processed_height
            ),
            "preprocessing": (
                "EXIF correction, RGB conversion, "
                "direct resize to 224x224, "
                "ImageNet normalization"
            ),
        },
        **analysis_result,
    }


@app.post("/api/analyze/custom")
async def custom_cam_analysis(
    image: UploadFile = File(...),
    mode: str = Form("quick"),
    selected_models: str = Form("[]"),
    selected_methods: str = Form("[]"),
    target_mode: str = Form("predicted"),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image format. "
                "Upload a JPG, JPEG, PNG, or WebP image."
            ),
        )

    image_bytes = await image.read()

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image is too large. Maximum upload size "
                f"is {MAX_UPLOAD_SIZE_MB} MB."
            ),
        )

    try:
        parsed_models = json.loads(
            selected_models
        )

        parsed_methods = json.loads(
            selected_methods
        )

        if not isinstance(
            parsed_models,
            list,
        ):
            raise ValueError(
                "selected_models must be a JSON list."
            )

        if not isinstance(
            parsed_methods,
            list,
        ):
            raise ValueError(
                "selected_methods must be a JSON list."
            )

        if mode not in {
            "quick",
            "full",
            "custom",
        }:
            raise ValueError(
                "mode must be quick, full, or custom."
            )

        if mode == "quick":
            methods_to_use = (
                QUICK_CAM_METHODS
            )

        elif mode == "full":
            methods_to_use = (
                FULL_CAM_METHODS
            )

        else:
            methods_to_use = (
                parsed_methods
            )

        models_to_use = (
            parsed_models
            if parsed_models
            else list(
                MODEL_CONFIGS.keys()
            )
        )

        processed_image = (
            process_uploaded_image(
                image_bytes
            )
        )

        analysis_result = (
            generate_cam_results(
                image_tensor=(
                    processed_image.tensor
                ),
                rgb_array=(
                    processed_image.rgb_array
                ),
                static_directory=STATIC_DIR,
                methods=methods_to_use,
                model_names=models_to_use,
                target_mode=target_mode,
            )
        )

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "selected_models and selected_methods "
                "must contain valid JSON arrays."
            ),
        ) from error

    except (
        ImageValidationError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    return {
        "filename": image.filename,
        "content_type": image.content_type,
        "mode": mode,
        "image_information": {
            "original_width": (
                processed_image.original_width
            ),
            "original_height": (
                processed_image.original_height
            ),
            "original_format": (
                processed_image.original_format
            ),
            "processed_width": (
                processed_image.processed_width
            ),
            "processed_height": (
                processed_image.processed_height
            ),
            "preprocessing": (
                "EXIF correction, RGB conversion, "
                "direct resize to 224x224, "
                "ImageNet normalization"
            ),
        },
        **analysis_result,
    }