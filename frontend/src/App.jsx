import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Eye,
  ImagePlus,
  Images,
  LoaderCircle,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from "lucide-react";

import {
  buildStaticUrl,
  fullAnalyzeImage,
  getHealthStatus,
  quickAnalyzeImage,
} from "./services/api";

import "./App.css";


const MAX_FILE_SIZE_MB = 15;

const SUPPORTED_FILE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
];

function formatPercentage(value) {
  if (typeof value !== "number") {
    return "0.00%";
  }

  return `${(value * 100).toFixed(2)}%`;
}


function formatTime(seconds) {
  if (typeof seconds !== "number") {
    return "0 ms";
  }

  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(1)} ms`;
  }

  return `${seconds.toFixed(3)} s`;
}


function PredictionBadge({ prediction }) {
  const isEdited = prediction === "edited";

  return (
    <span
      className={
        isEdited
          ? "prediction-badge prediction-edited"
          : "prediction-badge prediction-real"
      }
    >
      {isEdited ? "AI-Edited" : "Authentic"}
    </span>
  );
}


function ProbabilityBar({
  label,
  value,
}) {
  const percentage = Math.max(
    0,
    Math.min(100, value * 100)
  );

  return (
    <div className="probability-block">
      <div className="probability-header">
        <span>{label}</span>
        <strong>{formatPercentage(value)}</strong>
      </div>

      <div className="probability-track">
        <div
          className="probability-fill"
          style={{
            width: `${percentage}%`,
          }}
        />
      </div>
    </div>
  );
}


function ModelSummaryCard({
  result,
  isSelected,
  onSelect,
}) {
  const successfulCams =
    result.cam_results.filter(
      (cam) => cam.status === "success"
    ).length;

  return (
    <button
      type="button"
      className={
        isSelected
          ? "model-selector-card model-selector-active"
          : "model-selector-card"
      }
      onClick={onSelect}
    >
      <div className="model-selector-header">
        <div>
          <span>Backbone</span>
          <h3>{result.display_name}</h3>
        </div>

        <PredictionBadge
          prediction={result.prediction}
        />
      </div>

      <div className="model-selector-metrics">
        <div>
          <span>Confidence</span>
          <strong>
            {formatPercentage(result.confidence)}
          </strong>
        </div>

        <div>
          <span>CAMs ready</span>
          <strong>
            {successfulCams}/
            {result.cam_results.length}
          </strong>
        </div>
      </div>
    </button>
  );
}


function CamCard({
  cam,
  modelName,
}) {
  const [viewMode, setViewMode] =
    useState("overlay");

  const imageUrl =
    viewMode === "overlay"
      ? buildStaticUrl(cam.overlay_url)
      : buildStaticUrl(cam.heatmap_url);

  return (
    <article className="cam-card">
      <div className="cam-card-header">
        <div>
          <span className="cam-method-label">
            CAM method
          </span>

          <h3>{cam.display_name}</h3>
        </div>

        <div
          className={
            cam.status === "success"
              ? "cam-status cam-status-success"
              : "cam-status cam-status-error"
          }
        >
          {cam.status}
        </div>
      </div>

      {cam.warning && (
        <p className="cam-interpretation-note">
          {cam.warning}
        </p>
      )}

      {cam.status === "success" ? (
        <>
          <div className="cam-image-frame">
            <img
              src={imageUrl}
              alt={`${modelName} ${cam.display_name} ${viewMode}`}
            />
          </div>

          <div className="cam-view-toggle">
            <button
              type="button"
              className={
                viewMode === "overlay"
                  ? "toggle-button toggle-active"
                  : "toggle-button"
              }
              onClick={() =>
                setViewMode("overlay")
              }
            >
              <Images size={16} />
              Overlay
            </button>

            <button
              type="button"
              className={
                viewMode === "heatmap"
                  ? "toggle-button toggle-active"
                  : "toggle-button"
              }
              onClick={() =>
                setViewMode("heatmap")
              }
            >
              <Eye size={16} />
              Heatmap
            </button>
          </div>

          <div className="cam-runtime">
            <Clock3 size={16} />
            <span>Generation runtime</span>

            <strong>
              {formatTime(
                cam.runtime_seconds
              )}
            </strong>
          </div>
        </>
      ) : (
        <div className="cam-error">
          <AlertCircle size={20} />

          <div>
            <strong>
              Heatmap generation failed
            </strong>

            <p>{cam.error_message}</p>
          </div>
        </div>
      )}
    </article>
  );
}


function App() {
  const [analysisMode, setAnalysisMode] =
    useState("quick");

  const [
    showAuthenticAttention,
    setShowAuthenticAttention,
  ] = useState(false);

  const [selectedFile, setSelectedFile] =
    useState(null);

  const [previewUrl, setPreviewUrl] =
    useState("");

  const [analysisResult, setAnalysisResult] =
    useState(null);

  const [selectedModelName, setSelectedModelName] =
    useState("");

  const [isAnalyzing, setIsAnalyzing] =
    useState(false);

  const [errorMessage, setErrorMessage] =
    useState("");

  const [healthStatus, setHealthStatus] =
    useState(null);

  const [isDragging, setIsDragging] =
    useState(false);


  useEffect(() => {
    async function loadHealthStatus() {
      try {
        const status =
          await getHealthStatus();

        setHealthStatus(status);
      } catch {
        setHealthStatus(null);
      }
    }

    loadHealthStatus();
  }, []);


  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);


  const systemReady = useMemo(() => {
    return Boolean(
      healthStatus?.model_service?.loaded &&
      healthStatus?.all_checkpoints_available
    );
  }, [healthStatus]);


  const selectedModel = useMemo(() => {
    if (!analysisResult) {
      return null;
    }

    return (
      analysisResult.models.find(
        (model) =>
          model.model_name ===
          selectedModelName
      ) || analysisResult.models[0]
    );
  }, [
    analysisResult,
    selectedModelName,
  ]);


  function validateAndSelectFile(file) {
    setErrorMessage("");
    setAnalysisResult(null);
    setSelectedModelName("");

    if (!file) {
      return;
    }

    if (
      !SUPPORTED_FILE_TYPES.includes(
        file.type
      )
    ) {
      setErrorMessage(
        "Please upload a JPG, PNG, or WebP image."
      );

      return;
    }

    const maximumBytes =
      MAX_FILE_SIZE_MB * 1024 * 1024;

    if (file.size > maximumBytes) {
      setErrorMessage(
        `The image must be smaller than ${MAX_FILE_SIZE_MB} MB.`
      );

      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const newPreviewUrl =
      URL.createObjectURL(file);

    setSelectedFile(file);
    setPreviewUrl(newPreviewUrl);
  }


  function handleFileInput(event) {
    const file =
      event.target.files?.[0];

    validateAndSelectFile(file);
  }


  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);

    const file =
      event.dataTransfer.files?.[0];

    validateAndSelectFile(file);
  }


  function resetAnalysis() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(null);
    setPreviewUrl("");
    setAnalysisResult(null);
    setSelectedModelName("");
    setErrorMessage("");
    setShowAuthenticAttention(false);
  }


  async function handleAnalysis() {
    if (!selectedFile) {
      setErrorMessage(
        "Select an image before starting the analysis."
      );

      return;
    }

    setErrorMessage("");
    setIsAnalyzing(true);
    setAnalysisResult(null);
    setSelectedModelName("");
    setShowAuthenticAttention(false);

    try {
      const result =
        analysisMode === "full"
          ? await fullAnalyzeImage(
            selectedFile
          )
          : await quickAnalyzeImage(
            selectedFile
          );

      setAnalysisResult(result);

      setShowAuthenticAttention(
        result.consensus
          .default_show_heatmaps
      );

      if (result.models?.length > 0) {
        setSelectedModelName(
          result.models[0].model_name
        );
      }

    } catch (error) {
      const apiMessage =
        error?.response?.data?.detail;

      setErrorMessage(
        apiMessage ||
        "The image could not be analyzed."
      );

    } finally {
      setIsAnalyzing(false);
    }
  }


  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <BrainCircuit size={26} />
          </div>

          <div>
            <h1>
              AI-Edited Image Analyzer
            </h1>

            <p>
              Multi-backbone explainable
              image forensics
            </p>
          </div>
        </div>

        <div
          className={
            systemReady
              ? "system-status status-ready"
              : "system-status status-offline"
          }
        >
          {systemReady ? (
            <CheckCircle2 size={17} />
          ) : (
            <AlertCircle size={17} />
          )}

          <span>
            {systemReady
              ? "5 models ready"
              : "Backend unavailable"}
          </span>
        </div>
      </header>

      <main className="main-content">
        <section className="hero-section">
          <div className="hero-copy">
            <span className="eyebrow">
              Explainable image analysis
            </span>

            <h2>
              Detect possible AI editing and
              inspect the visual evidence
            </h2>

            <p>
              Quick analysis evaluates the
              uploaded image using five trained
              backbones and produces Grad-CAM,
              LayerCAM, and EigenCAM explanations
              for every model.
            </p>
          </div>

          <div className="hero-stat-grid">
            <div>
              <strong>5</strong>
              <span>Backbones</span>
            </div>

            <div>
              <strong>3</strong>
              <span>Quick CAM methods</span>
            </div>

            <div>
              <strong>15</strong>
              <span>Heatmaps per image</span>
            </div>
          </div>
        </section>

        <section className="workspace-grid">
          <div className="upload-panel">
            <div className="section-heading">
              <div>
                <span className="section-number">
                  01
                </span>

                <h2>Upload image</h2>
              </div>

              {selectedFile && (
                <button
                  type="button"
                  className="icon-button"
                  onClick={resetAnalysis}
                >
                  <X size={19} />
                </button>
              )}
            </div>

            {!previewUrl ? (
              <label
                className={
                  isDragging
                    ? "upload-zone upload-zone-dragging"
                    : "upload-zone"
                }
                onDrop={handleDrop}
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={(event) => {
                  event.preventDefault();
                  setIsDragging(false);
                }}
              >
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleFileInput}
                />

                <div className="upload-icon">
                  <ImagePlus size={34} />
                </div>

                <h3>
                  Drop an image here
                </h3>

                <p>
                  or click to browse from your
                  computer
                </p>

                <span>
                  JPG, PNG or WebP · Maximum
                  15 MB
                </span>
              </label>
            ) : (
              <div className="image-preview">
                <img
                  src={previewUrl}
                  alt="Selected upload preview"
                />

                <div className="preview-details">
                  <div>
                    <span>File name</span>
                    <strong>
                      {selectedFile?.name}
                    </strong>
                  </div>

                  <div>
                    <span>File size</span>
                    <strong>
                      {(
                        selectedFile.size /
                        1024 /
                        1024
                      ).toFixed(2)}
                      {" MB"}
                    </strong>
                  </div>
                </div>
              </div>
            )}

            {errorMessage && (
              <div className="error-message">
                <AlertCircle size={19} />
                <span>{errorMessage}</span>
              </div>
            )}

            <div className="analysis-mode-selector">
              <button
                type="button"
                className={
                  analysisMode === "quick"
                    ? "mode-option mode-option-active"
                    : "mode-option"
                }
                onClick={() =>
                  setAnalysisMode("quick")
                }
              >
                <strong>Quick analysis</strong>
                <span>5 models × 3 CAMs</span>
              </button>

              <button
                type="button"
                className={
                  analysisMode === "full"
                    ? "mode-option mode-option-active"
                    : "mode-option"
                }
                onClick={() =>
                  setAnalysisMode("full")
                }
              >
                <strong>Full analysis</strong>
                <span>5 models × 8 CAMs</span>
              </button>
            </div>

            {analysisMode === "full" && (
              <div className="full-mode-warning">
                <AlertCircle size={19} />

                <div>
                  <strong>
                    Full analysis is slower
                  </strong>

                  <p>
                    ScoreCAM and Ablation-CAM require
                    repeated model inference. ViT-B/16
                    can take more than ten seconds per
                    method.
                  </p>
                </div>
              </div>
            )}

            <button
              type="button"
              className="primary-button"
              disabled={
                !selectedFile ||
                isAnalyzing ||
                !systemReady
              }
              onClick={
                handleAnalysis
              }
            >
              {isAnalyzing ? (
                <>
                  <LoaderCircle
                    className="spinner"
                    size={20}
                  />
                  {analysisMode === "full"
                    ? "Generating 40 heatmaps..."
                    : "Generating 15 heatmaps..."}
                </>
              ) : (
                <>
                  <Sparkles size={20} />
                  {analysisMode === "full"
                    ? "Run full CAM analysis"
                    : "Run quick CAM analysis"}
                </>
              )}
            </button>

            <p className="accuracy-note">
              Predictions and confidence values
              describe this uploaded image.
              Accuracy and F1 values are reference
              scores measured on the AutoSplice
              test set.
            </p>
          </div>

          <aside className="information-panel">
            <div className="section-heading">
              <div>
                <span className="section-number">
                  02
                </span>

                <h2>Quick mode</h2>
              </div>
            </div>

            <ol className="pipeline-list">
              <li>
                <span>1</span>
                Correct image orientation
              </li>

              <li>
                <span>2</span>
                Convert to RGB
              </li>

              <li>
                <span>3</span>
                Resize to 224 × 224
              </li>

              <li>
                <span>4</span>
                Run five classifiers
              </li>

              <li>
                <span>5</span>
                Generate Grad-CAM
              </li>

              <li>
                <span>6</span>
                Generate LayerCAM
              </li>

              <li>
                <span>7</span>
                Generate EigenCAM
              </li>
            </ol>

            <div className="privacy-card">
              <ShieldCheck size={24} />

              <div>
                <h3>
                  Locally processed
                </h3>

                <p>
                  Images and heatmaps are handled
                  by the locally running FastAPI
                  and PyTorch backend.
                </p>
              </div>
            </div>
          </aside>
        </section>

        {isAnalyzing && (
          <section className="loading-card">
            <LoaderCircle
              className="spinner"
              size={32}
            />

            <div>
              <h2>
                Running explainable analysis
              </h2>

              <p>
                {analysisMode === "full"
                  ? "Five classifiers and forty CAM visualizations are being generated."
                  : "Five classifiers and fifteen CAM visualizations are being generated."}
              </p>
            </div>
          </section>
        )}

        {analysisResult && (
          <section className="results-section">
            <div className="results-heading">
              <div>
                <span className="eyebrow">
                  Analysis complete
                </span>

                <h2>
                  Classification and heatmap
                  results
                </h2>
              </div>

              <button
                type="button"
                className="secondary-button"
                onClick={resetAnalysis}
              >
                <RotateCcw size={18} />
                New image
              </button>
            </div>

            <div
              className={
                analysisResult.consensus
                  .prediction === "edited"
                  ? "consensus-card consensus-edited"
                  : "consensus-card consensus-real"
              }
            >
              <div>
                <span className="consensus-label">
                  Consensus prediction
                </span>

                <h3>
                  {analysisResult.consensus
                    .prediction === "edited"
                    ? "AI-Edited Image"
                    : "Authentic Image"}
                </h3>

                <p>
                  {
                    analysisResult.consensus
                      .agreement_votes
                  }
                  {" of "}
                  {
                    analysisResult.consensus
                      .total_models
                  }
                  {" models agreed."}
                </p>
              </div>

              <div className="agreement-score">
                <strong>
                  {formatPercentage(
                    analysisResult.consensus
                      .agreement_percentage
                  )}
                </strong>

                <span>Model agreement</span>
              </div>
            </div>

            <div
              className={
                analysisResult.consensus
                  .manipulation_detected
                  ? "interpretation-warning interpretation-edited"
                  : "interpretation-warning interpretation-authentic"
              }
            >
              <AlertCircle size={21} />

              <div>
                <strong>
                  {analysisResult.consensus
                    .manipulation_detected
                    ? "Suspected edit evidence"
                    : "Authentic prediction"}
                </strong>

                <p>
                  {
                    analysisResult.consensus
                      .warning
                  }
                </p>
              </div>
            </div>

            <div className="result-summary-grid">
              <div>
                <span>Edited votes</span>
                <strong>
                  {
                    analysisResult.consensus
                      .edited_votes
                  }
                </strong>
              </div>

              <div>
                <span>Authentic votes</span>
                <strong>
                  {
                    analysisResult.consensus
                      .real_votes
                  }
                </strong>
              </div>

              <div>
                <span>Total runtime</span>
                <strong>
                  {formatTime(
                    analysisResult
                      .total_runtime_seconds
                  )}
                </strong>
              </div>

              <div>
                <span>Generated CAMs</span>
                <strong>
                  {analysisResult.models.reduce(
                    (total, model) =>
                      total +
                      model.cam_results.filter(
                        (cam) =>
                          cam.status ===
                          "success"
                      ).length,
                    0
                  )}
                </strong>
              </div>
            </div>

            <section className="model-selection-section">
              <div className="subsection-heading">
                <div>
                  <span className="section-number">
                    03
                  </span>

                  <h2>
                    Select a backbone
                  </h2>
                </div>

                <p>
                  Choose a model to inspect its
                  prediction and heatmaps.
                </p>
              </div>

              <div className="model-selector-grid">
                {analysisResult.models.map(
                  (model) => (
                    <ModelSummaryCard
                      key={model.model_name}
                      result={model}
                      isSelected={
                        selectedModel
                          ?.model_name ===
                        model.model_name
                      }
                      onSelect={() =>
                        setSelectedModelName(
                          model.model_name
                        )
                      }
                    />
                  )
                )}
              </div>
            </section>

            {selectedModel && (
              <section className="selected-model-section">
                <div className="selected-model-header">
                  <div>
                    <span className="eyebrow">
                      Selected backbone
                    </span>

                    <h2>
                      {
                        selectedModel.display_name
                      }
                    </h2>
                  </div>

                  <PredictionBadge
                    prediction={
                      selectedModel.prediction
                    }
                  />
                </div>

                <div className="selected-model-stats">
                  <div>
                    <span>Confidence</span>
                    <strong>
                      {formatPercentage(
                        selectedModel.confidence
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>
                      AI-edited probability
                    </span>
                    <strong>
                      {formatPercentage(
                        selectedModel
                          .probability_edited
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>
                      Authentic probability
                    </span>
                    <strong>
                      {formatPercentage(
                        selectedModel
                          .probability_real
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>
                      Reference accuracy
                    </span>
                    <strong>
                      {formatPercentage(
                        selectedModel
                          .reference_test_accuracy
                      )}
                    </strong>
                  </div>
                </div>

                {!analysisResult.consensus
                  .manipulation_detected &&
                  !showAuthenticAttention && (
                    <div className="attention-hidden-card">
                      <ShieldCheck size={30} />

                      <div>
                        <h3>
                          Manipulation heatmaps are hidden
                        </h3>

                        <p>
                          Most models predicted that this
                          image is authentic. CAM outputs
                          may still be generated, but they
                          explain model attention and must
                          not be treated as manipulated
                          regions.
                        </p>

                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() =>
                            setShowAuthenticAttention(true)
                          }
                        >
                          <Eye size={18} />
                          Inspect model attention anyway
                        </button>
                      </div>
                    </div>
                  )}

                {(
                  analysisResult.consensus
                    .manipulation_detected ||
                  showAuthenticAttention
                ) && (
                  <>
                    <div className="original-comparison">
                  <div>
                    <span className="cam-method-label">
                      Processed model input
                    </span>

                    <h3>Original image</h3>

                    <div className="original-image-frame">
                      <img
                        src={buildStaticUrl(
                          analysisResult
                            .processed_image_url
                        )}
                        alt="Processed model input"
                      />
                    </div>
                  </div>

                  <div className="explanation-note">
                    <Eye size={26} />

                    <div>
                      <h3>
                        How to read the heatmaps
                      </h3>

                      <p>
                        Warmer regions indicate
                        areas that influenced the
                        edited-class explanation.
                        Heatmaps are not exact
                        segmentation masks and
                        should be interpreted
                        together with the model
                        probabilities.
                      </p>
                    </div>
                  </div>
                </div>

                    <div className="cam-grid">
                      {selectedModel.cam_results.map(
                        (cam) => (
                          <CamCard
                            key={cam.method}
                            cam={cam}
                            modelName={
                              selectedModel
                                .display_name
                            }
                          />
                        )
                      )}
                    </div>
                  </>
                )}
              </section>
            )}
          </section>
        )}
      </main>

      <footer className="footer">
        <p>
          Research prototype for explainable
          AI-edited image manipulation analysis.
        </p>
      </footer>
    </div>
  );
}


export default App;