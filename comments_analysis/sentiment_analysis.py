#!/usr/bin/env python3
"""
Sentiment Analysis Script using cardiffnlp/twitter-roberta-base-sentiment-latest

This script analyzes text sentiment using the RoBERTa model
pre-trained on Twitter for sentiment analysis.

The model classifies text into 3 categories:
- negative (0)
- neutral (1)
- positive (2)

Usage:
    python sentiment_analysis.py "Your text here"
    python sentiment_analysis.py --file input.txt
"""

import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import numpy as np
from scipy.special import softmax


class SentimentAnalyzer:
    """
    Sentiment analyzer using cardiffnlp/twitter-roberta-base-sentiment-latest
    """

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    LABELS = ["negative", "neutral", "positive"]

    def __init__(self, device: str = None):
        """
        Initializes the sentiment analysis model.

        Args:
            device: Device to run the model on ('cpu', 'cuda', 'mps').
                   If None, automatically chooses.
        """
        print(f"Loading model: {self.MODEL_NAME}...")

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.config = AutoConfig.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

    def preprocess(self, text: str) -> str:
        """
        Preprocesses text for the model (handles mentions and URLs).

        Args:
            text: Text to preprocess

        Returns:
            Preprocessed text
        """
        new_text = []
        for word in text.split(" "):
            word = '@user' if word.startswith('@') and len(word) > 1 else word
            word = 'http' if word.startswith('http') else word
            new_text.append(word)
        return " ".join(new_text)

    def analyze(self, text: str) -> dict:
        """
        Analyzes the sentiment of a text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with label, score, and probabilities for each class
        """
        # Preprocess
        processed_text = self.preprocess(text)

        # Tokenize
        encoded_input = self.tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Inference
        with torch.no_grad():
            output = self.model(**encoded_input)

        # Calculate probabilities
        scores = output.logits[0].cpu().numpy()
        scores = softmax(scores)

        # Find class with highest probability
        ranking = np.argsort(scores)[::-1]

        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "label": self.LABELS[ranking[0]],
            "confidence": float(scores[ranking[0]]),
            "probabilities": {
                self.LABELS[i]: float(scores[i]) for i in range(len(self.LABELS))
            },
            "ranking": [
                {"label": self.LABELS[i], "score": float(scores[i])}
                for i in ranking
            ]
        }

        return result

    def analyze_batch(self, texts: list, batch_size: int = 16) -> list:
        """
        Analyzes sentiment of multiple texts in batch.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size

        Returns:
            List of results
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            processed_batch = [self.preprocess(t) for t in batch]

            encoded_input = self.tokenizer(
                processed_batch,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                outputs = self.model(**encoded_input)

            scores_batch = outputs.logits.cpu().numpy()

            for j, (text, scores) in enumerate(zip(batch, scores_batch)):
                probs = softmax(scores)
                ranking = np.argsort(probs)[::-1]

                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "label": self.LABELS[ranking[0]],
                    "confidence": float(probs[ranking[0]]),
                    "probabilities": {
                        self.LABELS[k]: float(probs[k]) for k in range(len(self.LABELS))
                    }
                })

        return results


def format_result(result: dict) -> str:
    """Formats the result in a readable way."""
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"📝 Text: {result['text']}")
    output.append(f"{'='*60}")

    emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😞"}

    output.append(f"\n🎯 Sentiment: {emoji_map[result['label']]} {result['label'].upper()}")
    output.append(f"📊 Confidence: {result['confidence']*100:.2f}%")

    output.append(f"\n📈 Class probabilities:")
    for label, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        output.append(f"   {emoji_map[label]} {label:10s}: [{bar}] {prob*100:5.2f}%")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis with cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze"
    )
    parser.add_argument(
        "--file", "-f",
        help="Input file (one text per line)"
    )
    parser.add_argument(
        "--device", "-d",
        choices=["cpu", "cuda", "mps"],
        help="Device to use"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SentimentAnalyzer(device=args.device)

    if args.file:
        # Read texts from file
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        results = analyzer.analyze_batch(texts)

        if args.json:
            import json
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            for result in results:
                print(format_result(result))

    elif args.text:
        # Analyze single text
        result = analyzer.analyze(args.text)

        if args.json:
            import json
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(format_result(result))
    else:
        # Interactive mode
        print("\n🤖 Sentiment Analysis - Interactive Mode")
        print("Type a text and press Enter to analyze it.")
        print("Type 'quit' or 'exit' to quit.\n")

        while True:
            try:
                text = input("📝 Enter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                if not text:
                    continue

                result = analyzer.analyze(text)
                print(format_result(result))
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
