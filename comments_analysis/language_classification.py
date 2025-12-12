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
        "afr_Latn": "Afrikaans",
        "ara_Arab": "Arabic",
        "arb_Arab": "Arabic (Standard)",
        "ast_Latn": "Asturian",
        "azb_Arab": "South Azerbaijani",
        "azj_Latn": "North Azerbaijani",
        "ben_Beng": "Bengali",
        "bod_Tibt": "Tibetan",
        "bos_Latn": "Bosnian",
        "bul_Cyrl": "Bulgarian",
        "cat_Latn": "Catalan",
        "ces_Latn": "Czech",
        "cym_Latn": "Welsh",
        "dan_Latn": "Danish",
        "deu_Latn": "German",
        "ell_Grek": "Greek",
        "eng_Latn": "English",
        "epo_Latn": "Esperanto",
        "est_Latn": "Estonian",
        "eus_Latn": "Basque",
        "fas_Arab": "Persian",
        "fin_Latn": "Finnish",
        "fra_Latn": "French",
        "gle_Latn": "Irish",
        "glg_Latn": "Galician",
        "gom_Deva": "Goan Konkani",
        "heb_Hebr": "Hebrew",
        "hin_Deva": "Hindi",
        "hrv_Latn": "Croatian",
        "hun_Latn": "Hungarian",
        "ind_Latn": "Indonesian",
        "ita_Latn": "Italian",
        "jav_Latn": "Javanese",
        "jpn_Jpan": "Japanese",
        "kor_Hang": "Korean",
        "lav_Latn": "Latvian",
        "lit_Latn": "Lithuanian",
        "lmo_Latn": "Lombard",
        "ltz_Latn": "Luxembourgish",
        "lvs_Latn": "Standard Latvian",
        "mar_Deva": "Marathi",
        "mlt_Latn": "Maltese",
        "msa_Latn": "Malay",
        "nld_Latn": "Dutch",
        "nno_Latn": "Norwegian Nynorsk",
        "nob_Latn": "Norwegian Bokmål",
        "nor_Latn": "Norwegian",
        "oci_Latn": "Occitan",
        "pap_Latn": "Papiamento",
        "pol_Latn": "Polish",
        "por_Latn": "Portuguese",
        "roh_Latn": "Romansh",
        "ron_Latn": "Romanian",
        "rus_Cyrl": "Russian",
        "slk_Latn": "Slovak",
        "slv_Latn": "Slovenian",
        "spa_Latn": "Spanish",
        "srp_Cyrl": "Serbian",
        "swa_Latn": "Swahili",
        "swe_Latn": "Swedish",
        "tam_Taml": "Tamil",
        "tel_Telu": "Telugu",
        "tha_Thai": "Thai",
        "tur_Latn": "Turkish",
        "ukr_Cyrl": "Ukrainian",
        "urd_Arab": "Urdu",
        "vie_Latn": "Vietnamese",
        "war_Latn": "Waray",
        "xho_Latn": "Xhosa",
        "yor_Latn": "Yoruba",
        "yue_Hant": "Cantonese (Traditional)",
        "zho_Hans": "Chinese (Simplified)",
        "zho_Hant": "Chinese (Traditional)",
        "zsm_Latn": "Malaysian",
    }

    # Flag emojis for main languages
    LANGUAGE_FLAGS = {
        "afr_Latn": "🇿🇦",
        "ara_Arab": "🇸🇦",
        "arb_Arab": "🇸🇦",
        "ast_Latn": "🇪🇸",
        "azb_Arab": "🇮🇷",
        "azj_Latn": "🇦🇿",
        "ben_Beng": "🇧🇩",
        "bod_Tibt": "🇨🇳",
        "bos_Latn": "🇧🇦",
        "bul_Cyrl": "🇧🇬",
        "cat_Latn": "🇪🇸",
        "ces_Latn": "🇨🇿",
        "cym_Latn": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
        "dan_Latn": "🇩🇰",
        "deu_Latn": "🇩🇪",
        "ell_Grek": "🇬🇷",
        "eng_Latn": "🇬🇧",
        "epo_Latn": "🌍",
        "est_Latn": "🇪🇪",
        "eus_Latn": "🇪🇸",
        "fas_Arab": "🇮🇷",
        "fin_Latn": "🇫🇮",
        "fra_Latn": "🇫🇷",
        "gle_Latn": "🇮🇪",
        "glg_Latn": "🇪🇸",
        "gom_Deva": "🇮🇳",
        "heb_Hebr": "🇮🇱",
        "hin_Deva": "🇮🇳",
        "hrv_Latn": "🇭🇷",
        "hun_Latn": "🇭🇺",
        "ind_Latn": "🇮🇩",
        "ita_Latn": "🇮🇹",
        "jav_Latn": "🇮🇩",
        "jpn_Jpan": "🇯🇵",
        "kor_Hang": "🇰🇷",
        "lav_Latn": "🇱🇻",
        "lit_Latn": "🇱🇹",
        "lmo_Latn": "🇮🇹",
        "ltz_Latn": "🇱🇺",
        "lvs_Latn": "🇱🇻",
        "mar_Deva": "🇮🇳",
        "mlt_Latn": "🇲🇹",
        "msa_Latn": "🇲🇾",
        "nld_Latn": "🇳🇱",
        "nno_Latn": "🇳🇴",
        "nob_Latn": "🇳🇴",
        "nor_Latn": "🇳🇴",
        "oci_Latn": "🇫🇷",
        "pap_Latn": "🇨🇼",
        "pol_Latn": "🇵🇱",
        "por_Latn": "🇵🇹",
        "roh_Latn": "🇨🇭",
        "ron_Latn": "🇷🇴",
        "rus_Cyrl": "🇷🇺",
        "slk_Latn": "🇸🇰",
        "slv_Latn": "🇸🇮",
        "spa_Latn": "🇪🇸",
        "srp_Cyrl": "🇷🇸",
        "swa_Latn": "🇰🇪",
        "swe_Latn": "🇸🇪",
        "tam_Taml": "🇮🇳",
        "tel_Telu": "🇮🇳",
        "tha_Thai": "🇹🇭",
        "tur_Latn": "🇹🇷",
        "ukr_Cyrl": "🇺🇦",
        "urd_Arab": "🇵🇰",
        "vie_Latn": "🇻🇳",
        "war_Latn": "🇵🇭",
        "xho_Latn": "🇿🇦",
        "yor_Latn": "🇳🇬",
        "yue_Hant": "🇭🇰",
        "zho_Hans": "🇨🇳",
        "zho_Hant": "🇹🇼",
        "zsm_Latn": "🇲🇾",
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
