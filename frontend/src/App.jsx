import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (!selectedFile) return;

    if (
      !["image/png", "image/jpeg", "image/jpg"].includes(
        selectedFile.type
      )
    ) {
      setError("Please upload a PNG, JPG or JPEG image.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const analyzeImage = async () => {
    if (!file) {
      setError("Please select a chest X-ray image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data.error || "Prediction failed."
        );
      }

      setResult(data);
    } catch (err) {
      if (err.message.includes("chest X-ray")) {
        setError(err.message);
      } else {
        setError(
          "Unable to connect to the AI server. Make sure FastAPI is running on port 8000."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const removeImage = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
  };

  return (
    <div className="app">

      {/* =====================================================
          NAVBAR
      ====================================================== */}

      <header className="navbar">

        <div className="brand">

          <div className="brand-icon">
            ✚
          </div>

          <div>
            <h2>MedVision AI</h2>
            <span>Chest X-Ray Analysis</span>
          </div>

        </div>

        <div className="status">
          <span className="status-dot"></span>
          AI System Online
        </div>

      </header>


      {/* =====================================================
          MAIN
      ====================================================== */}

      <main className="container">

        {/* HERO */}

        <section className="hero">

          <div className="hero-badge">
            AI-POWERED MEDICAL IMAGING
          </div>

          <h1>
            Intelligent Chest X-Ray
            <span> Analysis</span>
          </h1>

          <p>
            Upload a chest X-ray image and let our DenseNet121
            deep learning model analyze it for pneumonia-related
            patterns.
          </p>

        </section>


        {/* =====================================================
            WORKSPACE
        ====================================================== */}

        <section className="workspace">


          {/* =================================================
              UPLOAD CARD
          ================================================== */}

          <div className="upload-card">

            <div className="section-heading">

              <div>

                <h3>
                  Upload Chest X-Ray
                </h3>

                <p>
                  Supported formats: PNG, JPG, JPEG
                </p>

              </div>

            </div>


            {/* NO IMAGE */}

            {!preview ? (

              <label className="drop-zone">

                <input
                  type="file"
                  accept=".png,.jpg,.jpeg"
                  onChange={handleFileChange}
                />

                <div className="upload-icon">
                  ↑
                </div>

                <h4>
                  Upload your X-ray image
                </h4>

                <p>
                  Click here to browse your computer
                  <br />
                  or select an image file
                </p>

                <span className="browse-button">
                  Choose Image
                </span>

              </label>

            ) : (

              /* IMAGE PREVIEW */

              <div className="preview-area">

                <img
                  src={preview}
                  alt="Chest X-ray preview"
                />

                <div className="file-info">

                  <strong>
                    {file.name}
                  </strong>

                  <span>
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>

                </div>

                <button
                  className="remove-button"
                  onClick={removeImage}
                >
                  Remove Image
                </button>

              </div>

            )}


            {/* ANALYZE BUTTON */}

            {preview && (

              <button
                className="analyze-button"
                onClick={analyzeImage}
                disabled={loading}
              >

                {loading ? (

                  <>
                    <span className="spinner"></span>
                    Analyzing X-Ray...
                  </>

                ) : (

                  <>
                    Analyze X-Ray →
                  </>

                )}

              </button>

            )}


            {/* ERROR */}

            {error && (

              <div className="error-message">

                <span>⚠</span>

                <div>
                  <strong>Unable to analyze image</strong>
                  <p>{error}</p>
                </div>

              </div>

            )}

          </div>


          {/* =================================================
              RESULT CARD
          ================================================== */}

          <div className="result-card">

            <div className="result-header">

              <div>

                <h3>
                  Analysis Result
                </h3>

                <p>
                  AI model prediction
                </p>

              </div>

              <div className="model-badge">
                DenseNet121
              </div>

            </div>


            {/* EMPTY RESULT */}

            {!result ? (

              <div className="empty-result">

                <div className="result-icon">
                  ◉
                </div>

                <h4>
                  No Analysis Yet
                </h4>

                <p>
                  Upload a chest X-ray and click
                  <br />
                  <strong>
                    Analyze X-Ray
                  </strong>
                  {" "}to see the result.
                </p>

              </div>

            ) : (

              /* =================================================
                 RESULT CONTENT
              ================================================== */

              <div className="result-content">


                {/* PREDICTION */}

                <div
                  className={`prediction ${
                    result.prediction === "PNEUMONIA"
                      ? "pneumonia"
                      : "normal"
                  }`}
                >

                  <span className="prediction-label">
                    Predicted Class
                  </span>

                  <strong>
                    {result.prediction}
                  </strong>

                </div>


                {/* PROBABILITY */}

                <div className="probability-section">

                  <div className="probability-header">

                    <span>
                      Pneumonia Probability
                    </span>

                    <strong>
                      {result.pneumonia_probability}%
                    </strong>

                  </div>

                  <div className="progress-bar">

                    <div
                      className="progress-fill"
                      style={{
                        width: `${result.pneumonia_probability}%`,
                      }}
                    ></div>

                  </div>

                </div>


                {/* CONFIDENCE */}

                <div className="confidence-box">

                  <span>
                    Model Confidence
                  </span>

                  <strong>
                    {result.confidence}%
                  </strong>

                </div>


                {/* MODEL INFORMATION */}

                <div className="model-info">

                  <div>

                    <span>
                      Model
                    </span>

                    <strong>
                      {result.model}
                    </strong>

                  </div>

                  <div>

                    <span>
                      Task
                    </span>

                    <strong>
                      Pneumonia Detection
                    </strong>

                  </div>

                </div>


                {/* =================================================
                    IMAGE EXPLANATION
                ================================================== */}

                {result.gradcam_available &&
                  result.gradcam_image && (

                    <div className="explanation-section">

                      <div className="explanation-heading">

                        <div>

                          <h3>
                            Explainable AI
                          </h3>

                          <p>
                            Grad-CAM visualization
                          </p>

                        </div>

                        <span className="xai-badge">
                          XAI
                        </span>

                      </div>


                      <div className="gradcam-container">

                        <img
                          src={`${API_URL}${result.gradcam_image}`}
                          alt="Grad-CAM explanation"
                          className="gradcam-image"
                        />

                      </div>


                      <p className="gradcam-description">

                        Highlighted regions represent areas
                        that contributed to the model's prediction.
                        This visualization is provided for research
                        and interpretability purposes.

                      </p>

                    </div>

                  )}


                {/* DISCLAIMER */}

                <div className="disclaimer">

                  <strong>
                    ⚠ Research Prototype
                  </strong>

                  <p>
                    This result is generated by an AI
                    research/decision-support system and is not
                    a clinical diagnosis. Model probability should
                    not be interpreted as medical certainty.
                  </p>

                </div>

              </div>

            )}

          </div>

        </section>

        {/* =====================================================
    MODEL PERFORMANCE
====================================================== */}

<section className="performance-section">

  <div className="performance-header">
    <div>
      <div className="hero-badge">MODEL EVALUATION</div>

      <h2>Model Performance</h2>

      <p>
        Evaluation results on the held-out test dataset.
      </p>
    </div>

    <div className="evaluation-badge">
      Test Set
    </div>
  </div>


  {/* METRICS */}

  <div className="metrics-grid">

    <div className="metric-card">
      <span className="metric-label">Accuracy</span>
      <strong>82%</strong>
      <small>Overall correctness</small>
    </div>

    <div className="metric-card">
      <span className="metric-label">Precision</span>
      <strong>65%</strong>
      <small>Pneumonia precision</small>
    </div>

    <div className="metric-card">
      <span className="metric-label">Recall</span>
      <strong>45%</strong>
      <small>Pneumonia recall</small>
    </div>

    <div className="metric-card">
      <span className="metric-label">F1 Score</span>
      <strong>53%</strong>
      <small>Precision-recall balance</small>
    </div>

    <div className="metric-card metric-highlight">
      <span className="metric-label">ROC-AUC</span>
      <strong>83.76%</strong>
      <small>Ranking performance</small>
    </div>

  </div>


  {/* EVALUATION DETAILS */}

  <div className="evaluation-grid">

    {/* CONFUSION MATRIX */}

    <div className="evaluation-card">

      <div className="evaluation-card-header">

        <div>
          <h3>Confusion Matrix</h3>

          <p>
            Test-set classification results
          </p>
        </div>

        <span>2,669 samples</span>

      </div>


      <div className="matrix-wrapper">

        <div className="matrix-axis-label predicted">
          Predicted
        </div>

        <div className="matrix">

          <div></div>

          <div className="matrix-label">
            NORMAL
          </div>

          <div className="matrix-label">
            PNEUMONIA
          </div>


          <div className="matrix-label vertical">
            NORMAL
          </div>

          <div className="matrix-cell correct">
            <strong>1921</strong>
            <span>True Negative</span>
          </div>

          <div className="matrix-cell incorrect">
            <strong>147</strong>
            <span>False Positive</span>
          </div>


          <div className="matrix-label vertical">
            PNEUMONIA
          </div>

          <div className="matrix-cell incorrect">
            <strong>330</strong>
            <span>False Negative</span>
          </div>

          <div className="matrix-cell correct">
            <strong>271</strong>
            <span>True Positive</span>
          </div>

        </div>

      </div>

    </div>


    {/* MODEL DETAILS */}

    <div className="evaluation-card">

      <div className="evaluation-card-header">

        <div>
          <h3>Model Details</h3>

          <p>
            Architecture and evaluation setup
          </p>
        </div>

      </div>


      <div className="details-list">

        <div>
          <span>Architecture</span>
          <strong>DenseNet121</strong>
        </div>

        <div>
          <span>Learning Approach</span>
          <strong>Transfer Learning</strong>
        </div>

        <div>
          <span>Input Size</span>
          <strong>224 × 224 × 3</strong>
        </div>

        <div>
          <span>Output</span>
          <strong>Binary Classification</strong>
        </div>

        <div>
          <span>Classes</span>
          <strong>NORMAL / PNEUMONIA</strong>
        </div>

        <div>
          <span>Evaluation Samples</span>
          <strong>2,669</strong>
        </div>

      </div>

    </div>

  </div>


  {/* INTERPRETATION */}

  <div className="performance-note">

    <div className="performance-note-icon">
      i
    </div>

    <div>
      <strong>Performance Interpretation</strong>

      <p>
        The model achieved an ROC-AUC of 83.76% on the held-out
        test set. Recall for the pneumonia class was 45%, meaning
        some pneumonia cases were not detected. These results
        demonstrate the research prototype's performance but do
        not establish clinical effectiveness.
      </p>
    </div>

  </div>

</section>


        {/* =====================================================
            FEATURES
        ====================================================== */}

        <section className="features">

          <div>

            <span>
              01
            </span>

            <h3>
              Deep Learning
            </h3>

            <p>
              DenseNet121 transfer learning architecture.
            </p>

          </div>


          <div>

            <span>
              02
            </span>

            <h3>
              Fast Analysis
            </h3>

            <p>
              Upload an X-ray and receive an AI prediction.
            </p>

          </div>


          <div>

            <span>
              03
            </span>

            <h3>
              Explainable AI
            </h3>

            <p>
              Grad-CAM highlights image regions
              associated with the prediction.
            </p>

          </div>

        </section>

      </main>


      {/* =====================================================
          FOOTER
      ====================================================== */}

      <footer>

        <p>
          MedVision AI • Research & Decision-Support Prototype
        </p>

      </footer>

    </div>
  );
}

export default App;