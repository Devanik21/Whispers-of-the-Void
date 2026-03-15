# Alien Signal Classification APP

![Language](https://img.shields.io/badge/Language-Jupyter%20Notebook-DA5B0B?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Alien-signal-classification-APP?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Alien-signal-classification-APP?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Signal intelligence from deep space — SETI-inspired classification of candidate technosignatures using ML.

---

**Topics:** `anomaly-detection` · `astrophysics-ml` · `deep-learning` · `machine-learning` · `neural-networks` · `radio-astronomy` · `radio-signal-classification` · `seti` · `signal-processing` · `cnn-spectrogram`

## Overview

This application implements a machine learning pipeline for classifying radio telescope observations as either Radio Frequency Interference (RFI) or candidate technosignatures — signals that exhibit properties inconsistent with known natural astrophysical or terrestrial sources. Inspired by real SETI (Search for Extraterrestrial Intelligence) pipelines, it extracts cadence, drift rate, signal-to-noise, and spectral occupancy features from simulated or real filterbank observations and runs them through a trained binary classifier.

The Streamlit interface provides an input panel for signal parameters derived from narrowband radio telescope data: drift rate (Hz/s), signal-to-noise ratio, frequency (MHz), hit count across multiple pointings, and spectral width. The classifier — trained on a labelled dataset of known RFI and simulated technosignature signals — returns a classification with confidence score and a cadence pattern visualisation.

The application also demonstrates the importance of the 'ON/OFF cadence test': a genuine technosignature candidate should appear in ON-source pointings and disappear in OFF-source pointings. The classifier incorporates cadence consistency as a key feature, following standard SETI analysis methodology.

---

## Motivation

As radio telescope arrays grow in sensitivity and data volume (MeerKAT, SKA, Parkes, GBT), automated candidate classification becomes essential. Manual inspection of millions of candidate signals per observation session is infeasible. This project demonstrates how ML can serve as a first-pass filter that dramatically reduces the candidate pool for human expert review, without discarding genuine anomalies.

---

## Architecture

```
Signal Parameters: drift_rate, SNR, freq_MHz, hit_count, ON/OFF cadence
        │
  Feature Engineering (log-SNR, drift_sign, cadence_ratio)
        │
  Binary Classifier (RF / Gradient Boost)
        │
  ┌─────────────────────┐
  │ RFI vs Candidate    │
  │ Confidence: 0.0–1.0 │
  └─────────────────────┘
        │
  Cadence Pattern Plot + Feature SHAP
```

---

## Features

### Signal Parameter Input
Precise input for drift rate (Hz/s), SNR, centre frequency (MHz), number of hits across pointings, and spectral width — matching standard SETI candidate parameterisation.

### Technosignature vs RFI Classification
Binary classifier outputs probability of the signal being a genuine candidate vs. terrestrial interference, with a confidence-calibrated score.

### Cadence Pattern Visualisation
Bar chart of signal presence/absence across ON-source and OFF-source telescope pointings, the gold-standard test for non-terrestrial signal origin.

### Feature Importance Chart
SHAP or feature importance bar chart showing which signal properties most strongly distinguish candidate signals from RFI in the training data.

### Frequency-Time Waterfall Simulation
Simulated waterfall plot visualising the signal's drift across the frequency axis over time, contextualising the drift rate parameter.

### Batch Signal File Processing
Upload a CSV of candidate signals from a telescope observation session for bulk classification and ranked candidate list output.

### Known RFI Catalogue Comparison
Cross-reference input signal parameters against a local RFI catalogue to flag signals matching known interference sources.

### Anomaly Score Display
For signals near the decision boundary, an anomaly score highlights edge cases that warrant human expert review regardless of classifier output.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **Streamlit** | Application UI | Clean radio astronomy interface |
| **scikit-learn** | ML pipeline | RandomForestClassifier, feature scaling, evaluation |
| **pandas** | Data handling | Candidate signal table management |
| **NumPy** | Feature engineering | Log-SNR, drift sign, cadence ratio computation |
| **Plotly** | Interactive visualisation | Waterfall plots, cadence bar charts, SHAP plots |
| **SHAP** | Explainability | Signal feature attribution for each classification |
| **Astropy (optional)** | Frequency unit conversion | MHz/GHz conversion and Doppler correction |

> **Key packages detected in this repo:** `scikit-learn` · `numpy` · `pandas` · `matplotlib` · `seaborn` · `scipy` · `xgboost` · `lightgbm` · `catboost` · `tensorflow`

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JS projects)
- `pip` or `npm` package manager
- Relevant API keys (see Configuration section)

### Installation

```bash
git clone https://github.com/Devanik21/Alien-signal-classification-APP.git
cd Alien-signal-classification-APP
python -m venv venv && source venv/bin/activate
pip install streamlit scikit-learn shap pandas numpy plotly
streamlit run app.py
```

---

## Usage

```bash
streamlit run app.py

# Batch classify candidates from a Breakthrough Listen CSV export
python classify_candidates.py --input bl_candidates.csv --output ranked.csv

# Retrain classifier with updated labelled data
python train.py --data labelled_signals.csv --model rf
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `signal_clf.pkl` | Trained binary classifier |
| `CADENCE_ON_COUNT` | `3` | Expected number of ON-source hits for genuine candidates |
| `RFI_CATALOGUE` | `known_rfi.csv` | Local RFI frequency catalogue for cross-matching |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence to label as genuine candidate |

> Copy `.env.example` to `.env` and populate all required values before running.

---

## Project Structure

```
Alien-signal-classification-APP/
├── README.md
├── requirements.txt
├── Advanced Insights.py
├── about.py
├── analyze.py
├── app.py
├── feedback.py
├── alien signal.ipynb
├── AI.json
├── narrowband signals.csv
└── ...
```

---

## Roadmap

- [ ] Integration with Breakthrough Listen Open Data Archive for live candidate querying
- [ ] Deep learning approach: 1D-CNN on raw filterbank spectra for end-to-end classification
- [ ] Multi-beam RFI rejection using simultaneous pointing correlation
- [ ] Real-time classification pipeline for streaming telescope data (Kafka + online inference)
- [ ] Candidate report generation in standard SETI format for community sharing

---

## Contributing

Contributions, issues, and feature requests are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please follow conventional commit messages and ensure any new code is documented.

---

## Notes

This is a research and educational tool. No claimed detections of extraterrestrial intelligence are made or implied. The classifier is trained on simulated data and known RFI patterns; performance on real telescope data requires careful validation.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Crafted with curiosity, precision, and a belief that good software is worth building well.*
