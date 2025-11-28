#!/usr/bin/env python3
"""
Language Classification Script using facebook/fasttext-language-identification

This script identifies the language of a text using the FastText model
from Facebook for language identification.

The model supports 217 languages.

Usage:
    python language_classification.py "Your text here"
    python language_classification.py --file input.txt
"""

import argparse
from huggingface_hub import hf_hub_download
import fasttext
import warnings

# Suppress fasttext warnings about numpy
warnings.filterwarnings('ignore')


class LanguageClassifier:
    """
    Language classifier using facebook/fasttext-language-identification
    """

    MODEL_NAME = "facebook/fasttext-language-identification"

    # Map language codes -> full names (main ones)
    LANGUAGE_NAMES = {
        "eng_Latn": "English",
        "ita_Latn": "Italian",
        "deu_Latn": "German",
        "fra_Latn": "French",
        "spa_Latn": "Spanish",
        "por_Latn": "Portuguese",
        "nld_Latn": "Dutch",
        "pol_Latn": "Polish",
        "rus_Cyrl": "Russian",
        "ukr_Cyrl": "Ukrainian",
        "zho_Hans": "Chinese (Simplified)",
        "zho_Hant": "Chinese (Traditional)",
        "jpn_Jpan": "Japanese",
        "kor_Hang": "Korean",
        "ara_Arab": "Arabic",
        "hin_Deva": "Hindi",
        "tur_Latn": "Turkish",
        "vie_Latn": "Vietnamese",
        "tha_Thai": "Thai",
        "heb_Hebr": "Hebrew",
        "ell_Grek": "Greek",
        "ces_Latn": "Czech",
        "ron_Latn": "Romanian",
        "hun_Latn": "Hungarian",
        "swe_Latn": "Swedish",
        "dan_Latn": "Danish",
        "nor_Latn": "Norwegian",
        "fin_Latn": "Finnish",
        "ind_Latn": "Indonesian",
        "msa_Latn": "Malay",
        "cat_Latn": "Catalan",
        "bul_Cyrl": "Bulgarian",
        "hrv_Latn": "Croatian",
        "slk_Latn": "Slovak",
        "slv_Latn": "Slovenian",
        "srp_Cyrl": "Serbian",
        "lit_Latn": "Lithuanian",
        "lav_Latn": "Latvian",
        "est_Latn": "Estonian",
        "ben_Beng": "Bengali",
        "tam_Taml": "Tamil",
        "tel_Telu": "Telugu",
        "mar_Deva": "Marathi",
        "urd_Arab": "Urdu",
        "fas_Arab": "Persian",
        "afr_Latn": "Afrikaans",
        "swa_Latn": "Swahili",
    }

    # Flag emojis for main languages
    LANGUAGE_FLAGS = {
        "eng_Latn": "🇬🇧",
        "ita_Latn": "🇮🇹",
        "deu_Latn": "🇩🇪",
        "fra_Latn": "🇫🇷",
        "spa_Latn": "🇪🇸",
        "por_Latn": "🇵🇹",
        "nld_Latn": "🇳🇱",
        "pol_Latn": "🇵🇱",
        "rus_Cyrl": "🇷🇺",
        "ukr_Cyrl": "🇺🇦",
        "zho_Hans": "🇨🇳",
        "zho_Hant": "🇹🇼",
        "jpn_Jpan": "🇯🇵",
        "kor_Hang": "🇰🇷",
        "ara_Arab": "🇸🇦",
        "hin_Deva": "🇮🇳",
        "tur_Latn": "🇹🇷",
        "vie_Latn": "🇻🇳",
        "tha_Thai": "🇹🇭",
        "heb_Hebr": "🇮🇱",
        "ell_Grek": "🇬🇷",
        "ces_Latn": "🇨🇿",
        "ron_Latn": "🇷🇴",
        "hun_Latn": "🇭🇺",
        "swe_Latn": "🇸🇪",
        "dan_Latn": "🇩🇰",
        "nor_Latn": "🇳🇴",
        "fin_Latn": "🇫🇮",
        "ind_Latn": "🇮🇩",
        "msa_Latn": "🇲🇾",
    }

    def __init__(self):
        """
        Initializes the language classifier.
        """
        print(f"Loading model: {self.MODEL_NAME}...")

        # Download and load FastText model
        model_path = hf_hub_download(self.MODEL_NAME, "model.bin")
        self.model = fasttext.load_model(model_path)

        print("Model loaded successfully!")

    def get_language_name(self, code: str) -> str:
        """Gets the full language name from the code."""
        return self.LANGUAGE_NAMES.get(code, code)

    def get_flag(self, code: str) -> str:
        """Gets the flag emoji for the language."""
        return self.LANGUAGE_FLAGS.get(code, "🏳️")

    def classify(self, text: str, top_k: int = 5) -> dict:
        """
        Classifies the language of a text.

        Args:
            text: Text to classify
            top_k: Number of languages to return in the ranking

        Returns:
            Dictionary with detected language and ranking
        """
        # Replace newlines with spaces (FastText requirement)
        text_cleaned = text.replace('\n', ' ').replace('\r', ' ')

        # Predict with FastText
        labels, scores = self.model.predict(text_cleaned, k=top_k)

        # Extract language codes (remove __label__ prefix)
        results = []
        for label, score in zip(labels, scores):
            lang_code = label.replace('__label__', '')
            results.append({
                "code": lang_code,
                "name": self.get_language_name(lang_code),
                "score": float(score),
                "flag": self.get_flag(lang_code)
            })

        top_result = results[0]

        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "language_code": top_result["code"],
            "language_name": top_result["name"],
            "confidence": top_result["score"],
            "flag": top_result["flag"],
            "ranking": results
        }

    def classify_batch(self, texts: list, top_k: int = 1) -> list:
        """
        Classifies the language of multiple texts in batch.

        Args:
            texts: List of texts to classify
            top_k: Number of languages to return for each text

        Returns:
            List of results
        """
        results = []

        for text in texts:
            # Clean text
            text_cleaned = text.replace('\n', ' ').replace('\r', ' ')

            # Predict
            labels, scores = self.model.predict(text_cleaned, k=top_k)

            # Extract first prediction
            lang_code = labels[0].replace('__label__', '')
            confidence = float(scores[0])

            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "language_code": lang_code,
                "language_name": self.get_language_name(lang_code),
                "confidence": confidence,
                "flag": self.get_flag(lang_code)
            })

        return results


def format_result(result: dict) -> str:
    """Formats the result in a readable way."""
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"📝 Text: {result['text']}")
    output.append(f"{'='*60}")

    output.append(f"\n🌍 Detected Language: {result['flag']} {result['language_name']}")
    output.append(f"🔤 Code: {result['language_code']}")
    output.append(f"📊 Confidence: {result['confidence']*100:.2f}%")

    if "ranking" in result and len(result["ranking"]) > 1:
        output.append(f"\n📈 Top candidate languages:")
        for r in result["ranking"]:
            bar_len = int(r["score"] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            output.append(f"   {r['flag']} {r['name']:15s}: [{bar}] {r['score']*100:5.2f}%")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Language Classification with facebook/fasttext-language-identification"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to classify"
    )
    parser.add_argument(
        "--file", "-f",
        help="Input file (one text per line)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of languages to show in ranking (default: 5)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format"
    )

    args = parser.parse_args()

    # Initialize classifier
    classifier = LanguageClassifier()

    if args.file:
        # Read texts from file
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        results = classifier.classify_batch(texts)

        if args.json:
            import json
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            for result in results:
                print(format_result(result))

    elif args.text:
        # Classify single text
        result = classifier.classify(args.text, top_k=args.top_k)

        if args.json:
            import json
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(format_result(result))
    else:
        # Interactive mode
        print("\n🌍 Language Classification - Interactive Mode")
        print("Type a text and press Enter to identify the language.")
        print("Type 'quit' or 'exit' to quit.\n")

        while True:
            try:
                text = input("📝 Enter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                if not text:
                    continue

                result = classifier.classify(text, top_k=args.top_k)
                print(format_result(result))
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
