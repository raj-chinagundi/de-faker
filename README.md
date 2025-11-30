# Fake Review Detector

A machine learning and AI-powered system to detect fake reviews using multiple approaches including traditional ML models, deep learning (LSTM/CNN), and LLMs (Gemini, Llama).

## Project Structure

```
de-faker/
├── app.py                          # Streamlit web application
├── prompts.py                      # Prompt templates for LLMs
├── requirements.txt                # Python dependencies
├── models/                         # Trained model artifacts
│   ├── logistic_regression_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── metadata_scaler.pkl
│   └── feature_info.pkl
├── src/                            # Source code and notebooks
│   ├── ml-code.ipynb               # ML models (Logistic Regression, Random Forest)
│   ├── Baseline_LSTM_Model.ipynb   # Baseline LSTM implementation
│   ├── Enchanced_LSTM_Model.ipynb  # Enhanced LSTM with features
│   ├── CNN_LSTM_Metadata_Model.ipynb # CNN-LSTM with metadata
│   ├── llama_main.py               # Llama model inference script
│   └── prompts.py                  # Prompt utilities
└── results/                        # Model outputs and visualizations
    ├── ML/
    ├── LSTM/
    ├── Llama-3.1-8b-Instruct/
    └── gemini-2.0-flash_result/
```

## Live Demo

The app is publicly available at: **https://de-faker.streamlit.app/**

## Installation

```bash
pip install -r requirements.txt
```

## Running the Streamlit App Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Running Source Code

### Machine Learning Models
```bash
jupyter notebook src/ml-code.ipynb
```

### LSTM Models
```bash
# Baseline LSTM
jupyter notebook src/Baseline_LSTM_Model.ipynb

# Enhanced LSTM
jupyter notebook src/Enchanced_LSTM_Model.ipynb

# CNN-LSTM with Metadata
jupyter notebook src/CNN_LSTM_Metadata_Model.ipynb
```

### Llama Model
```bash
python src/llama_main.py
```

---

Made with ❤️ for better review transparency
