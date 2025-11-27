"""
Fake Review Detection App
Streamlit UI for detecting fake reviews using ML or Gemini LLM
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide"
)

# ============================================================================
# FUNCTION DEFINITIONS (Must be defined before UI code)
# ============================================================================

@st.cache_resource
def load_ml_models():
    """Load and cache ML model artifacts."""
    model_dir = 'models'

    if not os.path.exists(f'{model_dir}/logistic_regression_model.pkl'):
        return None, None, None

    try:
        model = joblib.load(f'{model_dir}/logistic_regression_model.pkl')
        vectorizer = joblib.load(f'{model_dir}/tfidf_vectorizer.pkl')
        scaler = joblib.load(f'{model_dir}/metadata_scaler.pkl')

        # Verify vectorizer is fitted
        if not hasattr(vectorizer, 'idf_'):
            st.error("⚠️ Vectorizer is not fitted! Please retrain the model.")
            return None, None, None

        return model, vectorizer, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def predict_ml(review_text, rating, review_useful_count, review_length,
               friend_count, review_count, useful_count, cool_count,
               funny_count, compliment_count, tip_count, fan_count, restaurant_rating):
    """Make prediction using ML model."""

    # Load model artifacts (cached)
    model, vectorizer, scaler = load_ml_models()

    if model is None:
        return {
            'prediction': 'error',
            'message': '❌ Model not found! Please run train_model.py first to train the model.'
        }

    # Create feature vector
    # 1. TF-IDF features
    try:
        # Debug: Check if vectorizer has required attributes
        if not hasattr(vectorizer, 'idf_'):
            return {
                'prediction': 'error',
                'message': '❌ Vectorizer not properly fitted. Please retrain: python train_model.py'
            }

        text_features = vectorizer.transform([review_text])
    except Exception as e:
        return {
            'prediction': 'error',
            'message': f'❌ Vectorizer error: {str(e)}. Try retraining: python train_model.py'
        }

    # 2. Metadata features (same order as training)
    metadata = np.array([[
        friend_count,
        review_count,
        useful_count,
        cool_count,
        funny_count,
        compliment_count,
        tip_count,
        fan_count,
        rating,
        review_useful_count,
        review_length,
        restaurant_rating
    ]])

    # Scale metadata
    metadata_scaled = scaler.transform(metadata)

    # Combine features
    from scipy.sparse import hstack
    features = hstack([text_features, metadata_scaled])

    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return {
        'prediction': 'fake' if prediction == 1 else 'real',
        'confidence': float(probability[1]) if prediction == 1 else float(probability[0]),
        'probabilities': {
            'real': float(probability[0]),
            'fake': float(probability[1])
        },
        'method': 'Machine Learning (Logistic Regression)'
    }

def predict_gemini(review_text, rating, review_useful_count, review_length,
                   friend_count, review_count, useful_count, cool_count,
                   funny_count, compliment_count, restaurant_rating,
                   api_key, prompt_type):
    """Make prediction using Gemini LLM."""

    if not GEMINI_AVAILABLE:
        return {
            'prediction': 'error',
            'message': '❌ google-generativeai package not installed!'
        }

    # Import prompts
    try:
        from prompts import (
            format_zero_shot_prompt,
            format_few_shot_prompt,
            format_few_shot_cot_prompt
        )
    except ImportError:
        return {
            'prediction': 'error',
            'message': '❌ prompts.py not found! Please ensure it exists in the directory.'
        }

    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Prepare data dictionary
    review_data = {
        'reviewContent': review_text,
        'rating': rating,
        'reviewUsefulCount': review_useful_count,
        'ReviewLength': review_length,
        # 'date': datetime.now().strftime('%m/%d/%Y'),
        'reviewCount': review_count,
        'yelpJoinDate': 'N/A',
        'friendCount': friend_count,
        'usefulCount': useful_count,
        'coolCount': cool_count,
        'funnyCount': funny_count,
        'complimentCount': compliment_count,
        'restaurantRating': restaurant_rating
    }

    # Format prompt based on selection
    if prompt_type == "Zero-Shot":
        prompt = format_zero_shot_prompt(review_data)
    elif prompt_type == "Few-Shot":
        prompt = format_few_shot_prompt(review_data)
    else:  # Few-Shot with CoT
        prompt = format_few_shot_cot_prompt(review_data)

    # Generate response
    response = model.generate_content(prompt)

    # Parse JSON response
    try:
        # Extract JSON from response
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]

        result = json.loads(response_text)

        return {
            'prediction': result.get('flag', 'unknown'),
            'reasoning': result.get('reasoning', 'No reasoning provided'),
            'method': f'Gemini LLM ({prompt_type})',
            'raw_response': response.text
        }
    except json.JSONDecodeError:
        return {
            'prediction': 'error',
            'message': f'Failed to parse Gemini response. Raw response: {response.text}'
        }

def display_results(result):
    """Display prediction results in a beautiful format."""

    if result.get('prediction') == 'error':
        st.error(result.get('message', 'Unknown error occurred'))
        return

    prediction = result['prediction']
    method = result.get('method', 'Unknown')

    # Result header
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # Prediction box
    if prediction == 'fake':
        st.markdown(f"""
        <div class="result-box fake">
            <h2 style="color: #c53030; margin: 0;">⚠️ FAKE REVIEW DETECTED</h2>
            <p style="margin-top: 0.5rem; color: #742a2a;">This review shows indicators of being fraudulent.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box real">
            <h2 style="color: #2f855a; margin: 0;">✓ GENUINE REVIEW</h2>
            <p style="margin-top: 0.5rem; color: #22543d;">This review appears to be authentic.</p>
        </div>
        """, unsafe_allow_html=True)

    # Additional details
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Detection Method", method)

        if 'confidence' in result:
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")

    with col2:
        if 'probabilities' in result:
            st.write("**Probability Distribution:**")
            st.write(f"Real: {result['probabilities']['real']*100:.1f}%")
            st.write(f"Fake: {result['probabilities']['fake']*100:.1f}%")

    # Reasoning (for Gemini)
    if 'reasoning' in result:
        st.markdown("**Analysis Reasoning:**")
        st.info(result['reasoning'])

    # Raw response (for debugging)
    if 'raw_response' in result:
        with st.expander("🔍 View Raw LLM Response"):
            st.code(result['raw_response'], language='text')

# ============================================================================
# STREAMLIT UI (Functions must be defined above)
# ============================================================================

# Custom CSS for clean, modern design
st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Section headers with icons */
    .section-header {
        font-size: 0.9rem;
        font-weight: 600;
        color: #4A5568;
        margin-bottom: 1rem;
        margin-top: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        padding: 1rem;
        font-size: 0.95rem;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        padding: 0.5rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F7FAFC;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        font-weight: 500;
        padding: 0.75rem;
        font-size: 0.9rem;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #EDF2F7;
        border-color: #CBD5E0;
    }
    
    details[open] summary {
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    
    /* Button styling - purple gradient */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Result boxes */
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 5px solid;
    }
    
    .fake {
        background-color: #fff5f5;
        border-color: #fc8181;
    }
    
    .real {
        background-color: #f0fff4;
        border-color: #68d391;
    }
    
    /* Divider styling */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #E2E8F0;
    }
    
    /* Input labels */
    .stNumberInput label, .stSlider label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #4A5568;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for model selection and settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Model selection
    model_type = st.radio(
        "Detection Method",
        ["🤖 Machine Learning Model", "✨ Gemini LLM"],
        help="Choose between ML model or Gemini AI"
    )

    # Gemini settings
    if model_type == "✨ Gemini LLM":
        st.markdown("---")
        st.markdown("**Gemini Configuration**")

        gemini_api_key = st.text_input(
            "API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )

        prompt_type = st.selectbox(
            "Prompt Strategy",
            ["Zero-Shot", "Few-Shot", "Few-Shot with CoT"],
            help="Select the prompting approach"
        )

        if not GEMINI_AVAILABLE:
            st.warning("⚠️ google-generativeai not installed")



# Header - Clean and minimal
st.markdown('<h1 class="main-header">🔍 De-FakeR</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze reviews using Machine Learning or AI</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Main content area - Clean single column layout
col_label, col_count = st.columns([3, 1])
with col_label:
    st.markdown("📝 **Review Text**")
with col_count:
    review_length = 0
    st.markdown(f"<div style='text-align: right; color: #718096; font-size: 0.85rem;'>{review_length} characters</div>", unsafe_allow_html=True)

# Review text input
review_text = st.text_area(
    "Review Text",
    height=150,
    placeholder="Paste the review you want to analyze here...",
    help="The actual review content to analyze",
    label_visibility="collapsed",
    key="review_input"
)

# Update character count
review_length = len(review_text)

st.markdown("<br>", unsafe_allow_html=True)

# Collapsible metadata section with clean design
with st.expander("▼ Review Metadata (Optional)", expanded=False):
    
    # Review Information
    st.markdown('<div class="section-header">⭐ Review Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        rating = st.number_input("Rating (1-5)", 1, 5, 5, help="Star rating")
    with col2:
        review_useful_count = st.number_input("Review Useful Count", 0, 1000, 0, help="Useful votes")
    with col3:
        restaurant_rating = st.number_input("Restaurant Rating", 1.0, 5.0, 3.5, 0.5, help="Restaurant rating")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # User Profile
    st.markdown('<div class="section-header">👤 User Profile</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        friend_count = st.number_input("Friend Count", 0, 10000, 0, help="Number of friends")
    with col2:
        review_count = st.number_input("Review Count", 0, 10000, 1, help="Total reviews")
    with col3:
        fan_count = st.number_input("Fan Count", 0, 10000, 0, help="Number of fans")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # User Engagement
    st.markdown('<div class="section-header">👍 User Engagement</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        useful_count = st.number_input("Useful Count", 0, 100000, 0, help="Useful votes")
    with col2:
        cool_count = st.number_input("Cool Count", 0, 100000, 0, help="Cool votes")
    with col3:
        funny_count = st.number_input("Funny Count", 0, 100000, 0, help="Funny votes")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional Metrics
    st.markdown('<div class="section-header">📈 Additional Metrics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        compliment_count = st.number_input("Compliment Count", 0, 10000, 0, help="Total compliments")
    with col2:
        tip_count = st.number_input("Tip Count", 0, 10000, 0, help="Total tips")

# Prediction button - Full width with gradient
st.markdown("<br>", unsafe_allow_html=True)

if st.button("✨ Analyze Review", type="primary", use_container_width=True):
    if not review_text.strip():
        st.error("⚠️ Please enter a review to analyze!")
    else:
        with st.spinner("Analyzing review..."):
            try:
                if model_type == "🤖 Machine Learning Model":
                    # ML Model prediction
                    result = predict_ml(
                        review_text,
                        rating,
                        review_useful_count,
                        review_length,
                        friend_count,
                        review_count,
                        useful_count,
                        cool_count,
                        funny_count,
                        compliment_count,
                        tip_count,
                        fan_count,
                        restaurant_rating
                    )

                else:
                    # Gemini LLM prediction
                    if not gemini_api_key:
                        st.error("⚠️ Please enter your Gemini API key in the sidebar!")
                        st.stop()

                    result = predict_gemini(
                        review_text,
                        rating,
                        review_useful_count,
                        review_length,
                        friend_count,
                        review_count,
                        useful_count,
                        cool_count,
                        funny_count,
                        compliment_count,
                        restaurant_rating,
                        gemini_api_key,
                        prompt_type
                    )

                # Display results
                display_results(result)

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)
