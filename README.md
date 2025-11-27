# 🔍 Fake Review Detector

A Streamlit web application that detects fake reviews using Machine Learning (Logistic Regression) or Google's Gemini AI.

## ✨ Features

- **Dual Detection Methods**:
  - 🤖 Machine Learning (Logistic Regression with TF-IDF)
  - ✨ Gemini AI (Zero-Shot, Few-Shot, Few-Shot with CoT)
  
- **Comprehensive Analysis**:
  - Text-based features (TF-IDF vectorization)
  - Metadata features (user profile, engagement metrics)
  - Real-time predictions with confidence scores

- **Beautiful UI**:
  - Modern, clean design with purple gradient theme
  - Responsive layout
  - Collapsible metadata sections
  - Result visualization

## 🚀 Quick Start

### Local Installation

1. **Clone or download this directory**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the app**:
```bash
streamlit run app.py
```

4. **Open in browser**: The app will open at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

**Quick Deploy:**
1. Push this folder to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your repository
4. Deploy!

## 📊 Model Performance

The trained Logistic Regression model achieves:
- **Accuracy**: 86.31%
- **F1-Score**: 87.66%
- **Precision**: 88% (macro avg)
- **Recall**: 86% (macro avg)

## 🎯 How It Works

### Machine Learning Method

1. **Text Processing**: Reviews are vectorized using TF-IDF (5,000 features)
2. **Metadata Features**: 12 features including:
   - User profile (friend count, review count, fan count)
   - Engagement metrics (useful, cool, funny votes)
   - Review metadata (rating, length, restaurant rating)
3. **Classification**: Logistic Regression predicts Real/Fake with confidence

### Gemini AI Method

1. **Prompt Engineering**: Three strategies available:
   - Zero-Shot: Direct classification
   - Few-Shot: Learning from examples
   - Few-Shot with CoT: Chain-of-thought reasoning
2. **Analysis**: Gemini provides classification with reasoning

## 📁 Project Structure

```
de-fake/
├── app.py                 # Main Streamlit application
├── prompts.py            # Gemini prompt templates
├── train_model.py        # Model training script
├── test_models.py        # Model testing script
├── requirements.txt      # Python dependencies
├── models/               # Trained model files
│   ├── logistic_regression_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── metadata_scaler.pkl
│   └── feature_info.pkl
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── README.md            # This file
├── DEPLOYMENT.md        # Deployment guide
└── QUICKSTART.md        # Quick reference
```

## 🔑 Using Gemini AI

To use the Gemini detection method:

1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Select "✨ Gemini LLM" in the sidebar
3. Enter your API key
4. Choose a prompt strategy
5. Analyze reviews!

## 📝 Example Usage

### Analyzing a Review

1. **Paste Review Text**: Enter the review content
2. **Add Metadata (Optional)**: Expand the metadata section and fill in details
3. **Click Analyze**: Get instant results with confidence scores

### Sample Review for Testing

```
Amazing food best restaurant ever! The service was impeccable 
and the ambiance was perfect. Highly recommend to everyone! 
Five stars all the way!!!
```

Expected Result: **Fake** (extreme positive language, suspicious patterns)

## 🛠️ Retraining the Model

If you want to retrain with your own data:

1. Prepare your dataset in TSV format with required columns
2. Update paths in `train_model.py`
3. Run:
```bash
python train_model.py
```

The script will:
- Load and preprocess data
- Extract TF-IDF and metadata features
- Train Logistic Regression model
- Evaluate on test set
- Save model artifacts to `models/`

## 📈 Key Metadata Features

Research-backed indicators of fake reviews:

- **Useful Count**: 0-2 votes → 81.8% accuracy indicator
- **Friend Count**: ≤27 friends → 98.7% correlation with fake reviews
- **Review Count**: <30 reviews → 97% correlation with fake reviews
- **Rating**: Extreme ratings (1 or 5) → more common in fake reviews
- **Review Length**: <61 words → 65% correlation with fake reviews

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- ML powered by [scikit-learn](https://scikit-learn.org)
- AI powered by [Google Gemini](https://deepmind.google/technologies/gemini/)
- Dataset from Yelp reviews research

## 📞 Support

- **Issues**: Open an issue on GitHub
- **Questions**: Check [DEPLOYMENT.md](DEPLOYMENT.md) for troubleshooting
- **Updates**: Star the repo to get notified of updates

---

Made with ❤️ for better review transparency
