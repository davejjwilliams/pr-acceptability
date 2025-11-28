#!/usr/bin/env python3
"""
Gradio App for Sentiment Analysis and Language Classification

Simple web interface for text analysis using:
- cardiffnlp/twitter-roberta-base-sentiment-latest (Sentiment Analysis)
- facebook/fasttext-language-identification (Language Classification)

Usage:
    python app.py

The app will be accessible at http://localhost:7860
"""

import gradio as gr
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from huggingface_hub import hf_hub_download
import fasttext
import torch
import numpy as np
from scipy.special import softmax


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LANGUAGE_MODEL = "facebook/fasttext-language-identification"

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_EMOJIS = {"negative": "😞", "neutral": "😐", "positive": "😊"}

# Main language names
LANGUAGE_NAMES = {
    "eng_Latn": "English 🇬🇧",
    "ita_Latn": "Italian 🇮🇹",
    "deu_Latn": "German 🇩🇪",
    "fra_Latn": "French 🇫🇷",
    "spa_Latn": "Spanish 🇪🇸",
    "por_Latn": "Portuguese 🇵🇹",
    "nld_Latn": "Dutch 🇳🇱",
    "pol_Latn": "Polish 🇵🇱",
    "rus_Cyrl": "Russian 🇷🇺",
    "ukr_Cyrl": "Ukrainian 🇺🇦",
    "zho_Hans": "Chinese (Simplified) 🇨🇳",
    "zho_Hant": "Chinese (Traditional) 🇹🇼",
    "jpn_Jpan": "Japanese 🇯🇵",
    "kor_Hang": "Korean 🇰🇷",
    "ara_Arab": "Arabic 🇸🇦",
    "hin_Deva": "Hindi 🇮🇳",
    "tur_Latn": "Turkish 🇹🇷",
    "vie_Latn": "Vietnamese 🇻🇳",
    "tha_Thai": "Thai 🇹🇭",
    "heb_Hebr": "Hebrew 🇮🇱",
    "ell_Grek": "Greek 🇬🇷",
    "ces_Latn": "Czech 🇨🇿",
    "ron_Latn": "Romanian 🇷🇴",
    "hun_Latn": "Hungarian 🇭🇺",
    "swe_Latn": "Swedish 🇸🇪",
    "dan_Latn": "Danish 🇩🇰",
    "nor_Latn": "Norwegian 🇳🇴",
    "fin_Latn": "Finnish 🇫🇮",
}


# ============================================================
# MODEL LOADING (Lazy Loading)
# ============================================================

_sentiment_model = None
_sentiment_tokenizer = None
_language_model = None


def get_device():
    """Determines the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sentiment_model():
    """Loads the sentiment analysis model (lazy loading)."""
    global _sentiment_model, _sentiment_tokenizer

    if _sentiment_model is None:
        print("🔄 Loading sentiment analysis model...")
        device = get_device()

        _sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        _sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        _sentiment_model.to(device)
        _sentiment_model.eval()

        print(f"✅ Sentiment model loaded on {device}")

    return _sentiment_model, _sentiment_tokenizer


def load_language_model():
    """Loads the language classifier (lazy loading)."""
    global _language_model

    if _language_model is None:
        print("🔄 Loading language classification model...")

        model_path = hf_hub_download(LANGUAGE_MODEL, "model.bin")
        _language_model = fasttext.load_model(model_path)

        print(f"✅ Language model loaded")

    return _language_model


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def preprocess_sentiment(text: str) -> str:
    """Preprocesses text for sentiment (mentions and URLs)."""
    new_text = []
    for word in text.split(" "):
        word = '@user' if word.startswith('@') and len(word) > 1 else word
        word = 'http' if word.startswith('http') else word
        new_text.append(word)
    return " ".join(new_text)


def analyze_sentiment(text: str) -> str:
    """
    Analyzes the sentiment of the text.

    Returns:
        Formatted string with results
    """
    if not text or not text.strip():
        return "⚠️ Enter text to analyze."

    try:
        model, tokenizer = load_sentiment_model()
        device = get_device()

        # Preprocess
        processed_text = preprocess_sentiment(text)

        # Tokenize
        encoded_input = tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        # Inference
        with torch.no_grad():
            output = model(**encoded_input)

        # Calculate probabilities
        scores = output.logits[0].cpu().numpy()
        probs = softmax(scores)

        # Find main class
        top_idx = np.argmax(probs)
        top_label = SENTIMENT_LABELS[top_idx]
        top_score = probs[top_idx]

        # Format output
        result = []
        result.append(f"## 🎯 Sentiment Analysis Result\n")
        result.append(f"**Sentiment:** {SENTIMENT_EMOJIS[top_label]} **{top_label.upper()}**\n")
        result.append(f"**Confidence:** {top_score*100:.1f}%\n")
        result.append(f"\n### 📊 Probability Distribution\n")

        for i, label in enumerate(SENTIMENT_LABELS):
            emoji = SENTIMENT_EMOJIS[label]
            prob = probs[i]
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            result.append(f"- {emoji} **{label}**: `[{bar}]` {prob*100:.1f}%")

        return "\n".join(result)

    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"


def analyze_language(text: str) -> str:
    """
    Identifies the language of the text.

    Returns:
        Formatted string with results
    """
    if not text or not text.strip():
        return "⚠️ Enter text to analyze."

    try:
        model = load_language_model()

        # Clean text (FastText requirement)
        text_cleaned = text.replace('\n', ' ').replace('\r', ' ')

        # Predict with FastText
        labels, scores = model.predict(text_cleaned, k=5)

        # Extract results
        results = []
        for label, score in zip(labels, scores):
            lang_code = label.replace('__label__', '')
            results.append({
                "code": lang_code,
                "name": LANGUAGE_NAMES.get(lang_code, lang_code),
                "score": float(score)
            })

        top_result = results[0]

        # Format output
        result = []
        result.append(f"## 🌍 Language Classification Result\n")
        result.append(f"**Language:** {top_result['name']}\n")
        result.append(f"**Code:** `{top_result['code']}`\n")
        result.append(f"**Confidence:** {top_result['score']*100:.1f}%\n")
        result.append(f"\n### 📊 Top 5 Candidate Languages\n")

        for r in results:
            bar_len = int(r['score'] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            result.append(f"- **{r['name']}** (`{r['code']}`): `[{bar}]` {r['score']*100:.1f}%")

        return "\n".join(result)

    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"


def process_text(text: str, task: str) -> str:
    """
    Processes text based on selected task.

    Args:
        text: Text to analyze
        task: Task to execute ("Sentiment Analysis" or "Language Classification")

    Returns:
        Formatted result
    """
    if task == "Sentiment Analysis":
        return analyze_sentiment(text)
    elif task == "Language Classification":
        return analyze_language(text)
    else:
        return "⚠️ Select a valid task."


# ============================================================
# GRADIO INTERFACE
# ============================================================

def create_app():
    """Creates the Gradio interface."""

    with gr.Blocks(
        title="Text Analysis App"
    ) as app:

        gr.Markdown("""
        # 🔍 Text Analysis App

        Analyze text using Hugging Face models:
        - **Sentiment Analysis**: Classifies sentiment (positive/neutral/negative)
        - **Language Classification**: Identifies the language of the text
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Input
                text_input = gr.Textbox(
                    label="📝 Text to analyze",
                    placeholder="Enter text here...",
                    lines=5,
                    max_lines=10
                )

                # Task selection
                task_selector = gr.Dropdown(
                    choices=["Sentiment Analysis", "Language Classification"],
                    value="Sentiment Analysis",
                    label="🎯 Select Task"
                )

                # Button
                analyze_btn = gr.Button(
                    "🚀 Analyze",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=2):
                # Output
                result_output = gr.Markdown(
                    label="📊 Result",
                    value="*Results will appear here...*"
                )

        # Examples
        gr.Markdown("### 💡 Examples")

        with gr.Row():
            gr.Examples(
                examples=[
                    ["I love this product! It's amazing and works perfectly.", "Sentiment Analysis"],
                    ["This is terrible, worst experience ever.", "Sentiment Analysis"],
                    ["The meeting is scheduled for tomorrow at 3pm.", "Sentiment Analysis"],
                    ["Ciao, come stai? Oggi è una bella giornata!", "Language Classification"],
                    ["Bonjour le monde! Comment allez-vous aujourd'hui?", "Language Classification"],
                    ["今日は天気がいいですね。散歩に行きましょう。", "Language Classification"],
                    ["Hallo, wie geht es dir heute?", "Language Classification"],
                ],
                inputs=[text_input, task_selector],
                label="Click to try"
            )

        # Connect actions
        analyze_btn.click(
            fn=process_text,
            inputs=[text_input, task_selector],
            outputs=result_output
        )

        # Allow submit with Enter (Shift+Enter for new line)
        text_input.submit(
            fn=process_text,
            inputs=[text_input, task_selector],
            outputs=result_output
        )

        gr.Markdown("""
        ---
        ### ℹ️ Model Information

        | Task | Model | Description |
        |------|---------|-------------|
        | Sentiment Analysis | `cardiffnlp/twitter-roberta-base-sentiment-latest` | RoBERTa trained on Twitter, 3 classes |
        | Language Classification | `facebook/fasttext-language-identification` | Facebook FastText, 217 languages |

        *Models are loaded on first use (may take a few seconds).*
        """)

    return app


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("🚀 Starting Text Analysis App...")
    print()

    app = create_app()
    app.launch(share=True, debug=True)
