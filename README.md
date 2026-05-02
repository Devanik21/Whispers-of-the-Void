
# 🛸 Whispers of the Void

<img width="1672" height="941" alt="ChatGPT Image May 2, 2026, 09_52_30 AM" src="https://github.com/user-attachments/assets/638fe6cd-6d16-4fe6-b6e7-e73a69e44c29" />

> "Two possibilities exist: either we are alone in the Universe or we are not. Both are equally terrifying."
> — Arthur C. Clarke


---
A modular, Streamlit-based signal intelligence platform for **narrowband signal analysis, machine learning prediction, statistical exploration, advanced visualization, and Gemini-powered conversational insight**.

This repository is organized as a multi-page analytical system. The main router loads specialized pages for prediction, recommendation, visualization, analysis, insights, feedback, and AI-assisted exploration. :contentReference[oaicite:0]{index=0}

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why This Project Exists](#why-this-project-exists)
- [Core Capabilities](#core-capabilities)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Module Breakdown](#module-breakdown)
- [Data Schema](#data-schema)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Statistical Analysis Engine](#statistical-analysis-engine)
- [Visualization Stack](#visualization-stack)
- [Gemini AI Integration](#gemini-ai-integration)
- [Feedback System](#feedback-system)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Dependencies](#dependencies)
- [Usage Guide](#usage-guide)
- [Security Notes](#security-notes)
- [Performance Notes](#performance-notes)
- [Known Limitations](#known-limitations)
- [Future Enhancements](#future-enhancements)
- [Credits](#credits)
- [License](#license)

---

## Project Overview

The **Whispers of the Void** is a research-oriented Streamlit application built around a narrowband signal dataset. It combines interactive dashboards, classical machine learning, statistical diagnostics, and AI-assisted question answering into a single unified interface.

The app is intentionally modular. Each page focuses on one analytical responsibility, which makes the codebase easier to extend, debug, and repurpose for related signal-intelligence or astrophysical data analysis workflows. The central navigation shell in `app.py` routes users to pages such as Predict, Recommend, Visualize, Analyze, Insights, Feedback, CelestAI Nexus, and Advanced Insights. :contentReference[oaicite:1]{index=1}

---

## Why This Project Exists

This project is designed to support a full exploratory workflow for signal data:

- inspect raw data,
- visualize signal structure from many angles,
- filter candidate signals,
- run statistical tests,
- estimate feature relevance,
- train and evaluate basic classifiers,
- and ask a generative AI assistant conceptual questions about the signal domain.

The result is not just a dashboard, but a compact analytical workstation for signal interpretation.

---

## Core Capabilities

### 1. Signal Prediction
The prediction module loads a trained Random Forest model from `RF alien signal.pkl` and uses structured user inputs to classify a signal as safe or potentially anomalous. The user interface exposes sliders for the core signal features and displays both the raw prediction and a human-readable result. :contentReference[oaicite:2]{index=2}

### 2. Signal Recommendation and Filtering
The recommendation module filters the dataset based on frequency, duration, noise, and signal category constraints, then presents the resulting subset in tabular form and as interactive plots. It also allows CSV download of the filtered dataset. :contentReference[oaicite:3]{index=3}

### 3. Advanced Visualization
The visualization module provides a large collection of charts: bar plots, scatter plots, heatmaps, box plots, violin plots, pairplots, histograms, KDE plots, swarm plots, strip plots, joint plots, radar charts, treemaps, bubble charts, facet grids, hexbin plots, sunburst charts, and stacked area charts. :contentReference[oaicite:4]{index=4}

### 4. Statistical Analysis
The analysis modules support descriptive statistics, missing-value analysis, cleaning, feature engineering, custom filtering, grouping, aggregation, clustering, skewness estimation, outlier detection, correlation analysis, PCA, T-tests, chi-square tests, Random Forest feature importance, and RFE-based feature ranking. 

### 5. AI Assistance
The CelestAI Nexus page integrates Google Gemini and presents an AI question-answering interface for signal-related prompts, deep-space communication questions, and conceptual interpretation tasks. :contentReference[oaicite:6]{index=6}

### 6. Feedback Capture
The feedback page collects name, email, comments, a star rating, and consent, creating a simple structured user feedback loop. :contentReference[oaicite:7]{index=7}

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
├──────────────────────────────────────────────────────────┤
│  Page Router                                             │
│  - sidebar navigation                                    │
│  - dynamic page loading                                  │
│  - shared layout / branding                              │
├──────────────────────────────────────────────────────────┤
│  Analytical Modules                                      │
│  - predict.py                                            │
│  - recommend.py                                          │
│  - visualize.py                                          │
│  - analyze.py                                            │
│  - insights.py                                           │
│  - Advanced Insights.py                                  │
│  - CelestAI Nexus.py                                     │
│  - feedback.py                                           │
├──────────────────────────────────────────────────────────┤
│  Data / Model Layer                                      │
│  - narrowband signals.csv                                │
│  - RF alien signal.pkl                                   │
├──────────────────────────────────────────────────────────┤
│  External Service Layer                                  │
│  - Google Gemini API                                     │
└──────────────────────────────────────────────────────────┘
````

The main app file defines a `PAGES` mapping and loads the selected page script dynamically. This keeps the UI modular while maintaining a single entry point. 

---

## Repository Structure

```text
.
├── app.py
├── predict.py
├── recommend.py
├── visualize.py
├── analyze.py
├── insights.py
├── Advanced Insights.py
├── CelestAI Nexus.py
├── feedback.py
├── narrowband signals.csv
├── RF alien signal.pkl
├── AI.json
├── Designer.png
├── Designer2.png
└── README.md
```

---

## Module Breakdown

### `app.py`

The main navigation controller. It sets the page configuration, renders the shared header/footer, shows a sidebar image, and loads the selected page from the `PAGES` dictionary. The available pages include Predict, Recommend, Visualize, Analyze, Insights, Feedback, CelestAI Nexus, About, and Advanced Insights. 

### `predict.py`

The prediction page loads the model from `RF alien signal.pkl`, gathers user inputs through sliders, converts them into a one-row feature table, and runs inference. It then translates the model output into a friendly safe/alert classification message and visualizes the input values. 

### `recommend.py`

The recommendation page filters the dataset by frequency, duration, noise, and remark-based signal type. It displays filtered rows, computes summary statistics, offers CSV download, renders 2D and 3D scatter visualizations, and can train a classifier on the fly for signal labeling. 

### `visualize.py`

The visualization page is the most chart-heavy module. It reads the narrowband dataset and renders a large suite of plots, including correlation heatmaps, scatter plots, sunbursts, box plots, violin plots, pairplots, histograms, line plots, density maps, KDE plots, swarm plots, strip plots, joint plots, radar charts, treemaps, bubble charts, facet grids, hexbin plots, and stacked area charts. 

### `analyze.py`

The analysis page is a general-purpose data exploration tool. It allows a user to upload a CSV file, inspect shape and preview rows, display descriptive statistics, generate correlation heatmaps, analyze distributions, and run custom Pandas code. 

### `insights.py`

The insights page expands the analytical feature set. It includes descriptive statistics, missing data analysis, cleaning controls, feature creation, custom filters, aggregation, pairwise scatter comparison, clustering, processed-data export, skewness analysis, outlier detection, correlation analysis with p-values, hypothesis testing, PCA, Random Forest feature importance, model training, and summary insights. 

### `Advanced Insights.py`

This module adds another layer of analysis with column-wise distribution checks, skewness reporting, correlation heatmaps, outlier detection, PCA, Random Forest feature importance, custom visualizations, chi-square testing, target correlation, and recursive feature elimination. 

### `CelestAI Nexus.py`

The Gemini-integrated page configures the Google Generative AI client from a sidebar API key, loads a Lottie animation, shows example prompts, and lets the user ask questions that are passed to Gemini for generated responses. 

### `feedback.py`

The feedback page gathers name, email, feedback text, and a star rating, while also requiring consent before submission. It presents a preview of the submitted feedback as a DataFrame and includes contact information. 

---

## Data Schema

The app is built around the dataset `narrowband signals.csv`. Across the modules, the following columns are used repeatedly:

* `brightpixel`
* `narrowband`
* `narrowbanddrd`
* `noise`
* `Signal Frequency(MHz)`
* `Signal Duration(seconds)`
* `Stars Type`
* `Remarks`
* `Signal Origin `

These fields are used for classification, filtering, correlation analysis, feature importance, dimensionality reduction, plotting, and recommendation logic.

### Typical Roles of the Columns

* **`brightpixel`**: intensity-related signal feature
* **`narrowband`**: signal concentration feature
* **`narrowbanddrd`**: secondary narrowband descriptor
* **`noise`**: interference or contamination measure
* **`Signal Frequency(MHz)`**: signal frequency attribute
* **`Signal Duration(seconds)`**: signal duration attribute
* **`Stars Type`**: categorical grouping / label used in visualizations and models
* **`Remarks`**: class or annotation field used for filtering and prediction logic
* **`Signal Origin `**: additional categorical/numeric origin descriptor used by the prediction page 

---

## Machine Learning Pipeline

The project uses classical supervised learning and exploratory ML techniques.

### Prediction Workflow

The prediction page loads a persisted Random Forest model with `joblib`, builds a structured feature table from user sliders, and performs inference. The returned class is mapped to a safe/alert interpretation. 

### Training in Exploratory Pages

Several pages train models dynamically during runtime:

* Random Forest feature importance analysis,
* Random Forest classification on selected features,
* K-Means clustering,
* Logistic Regression-based RFE.

These are exploratory workflows rather than a single production pipeline.

### Core ML Components

* **Random Forest Classifier**
* **K-Means**
* **PCA**
* **RFE**
* **Label Encoding**
* **Classification Report generation**

---

## Statistical Analysis Engine

The statistical layer is broad and deliberately exploratory.

### Implemented Analyses

* descriptive statistics
* missing-value counts
* skewness estimation
* correlation matrices
* Pearson p-values
* T-tests
* chi-square tests
* outlier detection using Z-scores
* PCA decomposition
* feature importance scoring
* feature ranking via RFE

### Analytical Intent

The app is designed to help users move from raw values to structural understanding:

1. What does the dataset look like?
2. Which variables are correlated?
3. Which features are unusual?
4. Which features matter most?
5. Which signal groups differ statistically?
6. How can the signal space be visualized in lower dimensions?

---

## Visualization Stack

The repository uses both Matplotlib/Seaborn and Plotly to cover static and interactive use cases.

### Seaborn / Matplotlib

Used for:

* heatmaps
* scatter plots
* box plots
* violin plots
* histograms
* KDE plots
* swarm plots
* strip plots
* joint plots
* facet grids

### Plotly

Used for:

* sunburst plots
* radar charts
* treemaps
* bubble charts
* stacked area charts
* interactive scatter plots
* 3D scatter charts
* PCA scatter plots

The visualization page demonstrates an intentionally dense plotting strategy so the same signal data can be inspected from many perspectives. 

---

## Gemini AI Integration

The `CelestAI Nexus.py` page integrates Google Gemini through `google.generativeai`. A user enters an API key in the sidebar, the app configures the Gemini client, and the text prompt is sent to the model for generated analysis. The page also includes example prompts and a Lottie animation for presentation quality. 

### Example Use Cases

* explain signal anomalies,
* describe conceptual aspects of extraterrestrial communication,
* assist with interpretation of features,
* answer exploratory questions in plain language.

---

## Feedback System

The feedback module is intentionally lightweight but structured. It captures:

* name,
* email,
* free-text feedback,
* star rating,
* consent confirmation.

On successful submission, it displays a preview table of the collected information. 

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Ensure required files are present

The app expects:

* `narrowband signals.csv`
* `RF alien signal.pkl`
* image assets such as `Designer.png` and `Designer2.png`
* `AI.json` for the Lottie animation on the Gemini page

---

## Running the App

If the main entry file is `app.py`:

```bash
streamlit run app.py
```

If you rename it to `app.py` for cleanup:

```bash
streamlit run app.py
```

---

## Dependencies

A typical environment for this repository will include:

* `streamlit`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `plotly`
* `scipy`
* `statsmodels`
* `scikit-learn`
* `joblib`
* `google-generativeai`
* `streamlit-lottie`
* `requests`

Optional or standard-library modules used in the code:

* `json`

---

## Usage Guide

### Prediction

Open the Predict page, adjust sliders, and click the prediction button. The result is shown as a human-readable classification. 

### Recommendation

Open the Recommend page, set the filter ranges, view the subset, and download the filtered CSV if needed. 

### Visualization

Open the Visualize page and choose among the many chart types to inspect structure, density, composition, and relationships. 

### Analysis

Upload a CSV to the Analyze page, review summary statistics, and use custom Pandas expressions for deeper inspection. 

### Insights

Use the Insights page for advanced exploratory analysis, clustering, PCA, feature importance, outlier detection, and statistical tests. 

### Gemini Assistant

Open CelestAI Nexus, enter a Gemini API key, and ask questions about the signal domain. 

### Feedback

Submit your name, email, feedback, and rating from the Feedback page. 

---

## Security Notes

This repository includes a few patterns that are powerful but should be handled carefully in production:

### 1. Dynamic page execution

The main router loads page files dynamically using `exec(...)`. This is convenient for prototyping but should be replaced with a safer import-based mechanism in production. 

### 2. Custom code execution

The Analyze page allows user-provided Pandas code to be executed with `eval(...)`. This is useful for advanced experimentation but is unsafe for untrusted users. 

### 3. External API access

The Gemini page requires an API key. That key should be protected through Streamlit secrets or environment variables rather than hard-coded. 

### 4. File dependencies

Several pages assume the presence of exact filenames for the CSV, model, and image/animation assets. Missing files will cause runtime errors unless handled explicitly.

---

## Performance Notes

Some pages are computationally heavy by design.

### Potential Bottlenecks

* repeated CSV loading,
* large pairplots and facet grids,
* ad hoc model training during reruns,
* expensive Plotly visualizations on large datasets.

### Recommended Improvements

* cache CSV loading,
* cache model loading,
* avoid retraining during every rerun,
* sample large datasets before rendering dense plots,
* centralize shared preprocessing.

---

## Known Limitations

* The application depends on exact column names.
* Several sections assume the dataset has all required fields.
* Some analytics are exploratory rather than production-hardened.
* The main router uses a prototype-style dynamic loader.
* The custom Pandas code execution path is not secure for public deployment.
* Some visualization sections may become heavy on large datasets.

---

## Future Enhancements

### Recommended Technical Upgrades

* Replace `exec()` with clean import-based page routing.
* Add a shared preprocessing utility layer.
* Add validation for column names and data types.
* Cache data/model loading with Streamlit decorators.
* Split large pages into reusable components.
* Wrap analysis operations in safer helper functions.
* Introduce an exportable report generator.

### Advanced Research Extensions

* spectrogram-based deep learning,
* anomaly detection without labels,
* probabilistic confidence scoring,
* real-time signal stream ingestion,
* automated model comparison,
* stronger astrophysical context integration,
* experiment tracking and reproducibility tooling.

---

## Credits

Built with:

* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* SciPy
* Statsmodels
* Scikit-learn
* Google Generative AI
* Streamlit Lottie

The repository architecture and implemented behaviors are reflected directly in the source modules: main routing in `app.py`, visualization in `visualize.py`, analysis in `analyze.py`, advanced insight generation in `insights.py` and `Advanced Insights.py`, prediction in `predict.py`, recommendations in `recommend.py`, Gemini-based assistant functionality in `CelestAI Nexus.py`, and user feedback capture in `feedback.py`.

---

## License

Choose the license that matches your intended distribution model:

* MIT
* Apache 2.0
* GPL-3.0
* Proprietary

---

## Final Note

This repository is structured like a compact **signal intelligence lab**: it combines prediction, visualization, statistical inference, AI reasoning, and user feedback inside one modular Streamlit application. Its strength is breadth: the same dataset can be explored as a classification problem, a visualization problem, a statistical problem, and a conversational AI problem.




