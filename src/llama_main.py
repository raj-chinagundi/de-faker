import pandas as pd
import json
import os
import re
from pathlib import Path
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from prompts import get_prompt_formatter


# Configuration
VLLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
RESULTS_DIR = Path("./results")


def parse_llm_response(response_text):
    """
    Parse LLM response with robust edge case handling.
    
    Args:
        response_text: Raw text response from LLM
        
    Returns:
        dict: Parsed response with 'flag' and 'reasoning' keys
    """
    try:
        response_text = response_text.strip()
        
        # Try direct JSON parsing
        if response_text.startswith('{') and response_text.endswith('}'):
            parsed = json.loads(response_text)
            flag = parsed.get('flag', '').lower()
            if flag in ['fake', 'real']:
                return {'flag': flag, 'reasoning': parsed.get('reasoning', 'No reasoning provided')}
        
        # Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
            flag = parsed.get('flag', '').lower()
            if flag in ['fake', 'real']:
                return {'flag': flag, 'reasoning': parsed.get('reasoning', 'No reasoning provided')}
        
        # Extract JSON object anywhere in text
        json_match = re.search(r'\{[^}]*"flag"[^}]*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            flag = parsed.get('flag', '').lower()
            if flag in ['fake', 'real']:
                return {'flag': flag, 'reasoning': parsed.get('reasoning', 'No reasoning provided')}
        
        # Fallback: keyword detection
        response_lower = response_text.lower()
        if 'fake' in response_lower[:100]:
            return {'flag': 'fake', 'reasoning': 'Parsed from keyword detection'}
        elif 'real' in response_lower[:100]:
            return {'flag': 'real', 'reasoning': 'Parsed from keyword detection'}
        
        # Default to 'real' if parsing fails
        return {'flag': 'real', 'reasoning': 'Parsing failed - default classification'}
        
    except Exception as e:
        return {'flag': 'real', 'reasoning': f'Error during parsing: {str(e)}'}


def call_llm(prompt, client):
    """
    Call LLM via vLLM OpenAI-compatible API.
    
    Args:
        prompt: Formatted prompt string
        client: OpenAI client instance
        
    Returns:
        str: Raw response text from model
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512
    )
    return response.choices[0].message.content


def evaluate_predictions(y_true, y_pred, y_scores, run_type):
    """
    Calculate all evaluation metrics and save visualizations.
    
    Args:
        y_true: True binary labels (1=fake, 0=real)
        y_pred: Predicted binary labels
        y_scores: Prediction scores for ROC/PR curves
        run_type: Type of run (zero_shot, few_shot, few_shot_cot)
        
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    metrics['auc_roc'] = auc(fpr, tpr)
    
    # Precision-Recall and AUC-PR
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    metrics['auc_pr'] = average_precision_score(y_true, y_scores)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Save visualizations
    output_dir = RESULTS_DIR / run_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc_roc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {run_type}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f'PR Curve (AP = {metrics["auc_pr"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {run_type}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {run_type}')
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


def run_evaluation(df, run_type='zero_shot'):
    """
    Run evaluation pipeline for specified prompt type.
    
    Args:
        df: DataFrame with review data
        run_type: One of 'zero_shot', 'few_shot', 'few_shot_cot'
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Running {run_type.upper().replace('_', '-')} evaluation")
    print(f"{'='*60}\n")
    
    # Initialize client
    client = OpenAI(api_key="EMPTY", base_url=VLLM_API_BASE)
    
    # Get prompt formatter
    formatter = get_prompt_formatter(run_type)
    
    # Storage
    predictions = []
    true_labels = []
    prediction_scores = []
    results_data = []
    
    # Process each review
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {run_type}"):
        # Format prompt
        prompt = formatter(row)
        
        # Get LLM response
        response_text = call_llm(prompt, client)
        
        # Parse response
        parsed = parse_llm_response(response_text)
        
        # Convert to binary (1=fake, 0=real)
        pred_binary = 1 if parsed['flag'] == 'fake' else 0
        true_binary = int(row['flagged'])
        
        predictions.append(pred_binary)
        true_labels.append(true_binary)
        prediction_scores.append(pred_binary)
        
        results_data.append({
            'review_id': row.get('reviewID', idx),
            'true_label': 'fake' if true_binary == 1 else 'real',
            'predicted_label': parsed['flag'],
            'reasoning': parsed['reasoning'],
            'correct': pred_binary == true_binary
        })
    
    # Calculate metrics
    metrics = evaluate_predictions(true_labels, predictions, prediction_scores, run_type)
    
    # Save results
    output_dir = RESULTS_DIR / run_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to JSON
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed predictions
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Print metrics
    print(f"\n{'='*60}")
    print(f"RESULTS - {run_type.upper().replace('_', '-')}")
    print(f"{'='*60}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"MCC:         {metrics['mcc']:.4f}")
    print(f"AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:      {metrics['auc_pr']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={metrics['confusion_matrix'][0][0]}, FP={metrics['confusion_matrix'][0][1]}],")
    print(f"   [FN={metrics['confusion_matrix'][1][0]}, TP={metrics['confusion_matrix'][1][1]}]]")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return metrics


def check_gpu_availability():
    """Check and print GPU availability information."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ No GPU detected - vLLM will use CPU (very slow)")
            print("  Consider using --sample_size for testing or use a GPU instance")
        return cuda_available
    except ImportError:
        print("⚠ PyTorch not found - cannot check GPU status")
        return None


def main():
    """Main execution function with command-line flags."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fake Review Detection with LLM')
    parser.add_argument('--run_type', type=str, choices=['zero_shot', 'few_shot', 'few_shot_cot', 'all'],
                        default='zero_shot', help='Type of prompt to use')
    parser.add_argument('--data_path', type=str, default='./fake-review-detection/new_data_train.csv',
                        help='Path to CSV data file')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    # Check GPU availability
    print("\n" + "="*60)
    print("SYSTEM CHECK")
    print("="*60)
    check_gpu_availability()
    print("="*60 + "\n")
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path, encoding='utf-8', delimiter='\t')
    
    # Sample if requested
    if args.sample_size:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
        print(f"Using sample of {len(df)} reviews")
    else:
        print(f"Using all {len(df)} reviews")
    
    # Run evaluation(s)
    if args.run_type == 'all':
        for run_type in ['zero_shot', 'few_shot', 'few_shot_cot']:
            run_evaluation(df, run_type)
    else:
        run_evaluation(df, args.run_type)


if __name__ == "__main__":
    main()