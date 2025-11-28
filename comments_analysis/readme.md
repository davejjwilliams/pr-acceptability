# 🔍 Comments Analysis Tools

Text analysis tools for sentiment analysis and language classification of Pull Request comments and descriptions.

## 📦 Installation

```bash
# Navigate to the comments_analysis directory
cd comments_analysis

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install transformers torch scipy numpy gradio
```

## 🚀 Usage

### 1. Gradio App (Web Interface) - RECOMMENDED

```bash
python app.py
```

Open your browser at `http://localhost:7860`

**Features:**
- 📝 Text input field
- 🎯 Dropdown menu to choose the task
- 🚀 Button to run analysis
- 📊 Formatted results with confidence scores

### 2. Sentiment Analysis (CLI)

```bash
# Single text
python sentiment_analysis.py "I love this product!"

# From file
python sentiment_analysis.py --file texts.txt

# JSON output
python sentiment_analysis.py "Great work!" --json

# Interactive mode
python sentiment_analysis.py
```

### 3. Language Classification (CLI)

```bash
# Single text
python language_classification.py "Ciao, come stai?"

# From file
python language_classification.py --file texts.txt

# Top K languages
python language_classification.py "Hallo Welt" --top-k 10

# JSON output
python language_classification.py "Bonjour!" --json

# Interactive mode
python language_classification.py
```

## 📊 AIDev Dataset

To load the dataset:

```python
import pandas as pd

# Load PR data
pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet")

# Load PR comments
comments_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_comments.parquet")

# Load PR reviews
reviews_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_reviews.parquet")

# Load PR review comments
review_comments_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_review_comments_v2.parquet")
```

### Key Columns for Analysis

**For PR acceptance:**
- `merged_at IS NOT NULL` → PR Accepted
- `merged_at IS NULL AND state='closed'` → PR Rejected

**For Sentiment Analysis:**
- `title` - PR title
- `body` - PR description
- `body` (comments) - Comment text
- `body` (reviews) - Review text

**For Language Classification:**
- `title` - Identify language of title
- `body` - Identify language of description or comment

## 🤖 Models Used

| Task | Model | Link |
|------|---------|------|
| Sentiment Analysis | cardiffnlp/twitter-roberta-base-sentiment-latest | [HuggingFace](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) |
| Language Classification | facebook/fasttext-language-identification | [HuggingFace](https://huggingface.co/facebook/fasttext-language-identification) |

## 💡 Analysis Example on AIDev

```python
import pandas as pd
from sentiment_analysis import SentimentAnalyzer
from language_classification import LanguageClassifier

# Load data
pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet")
comments_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_comments.parquet")

# Initialize models
sentiment = SentimentAnalyzer()
language = LanguageClassifier()

# Analyze a sample
sample_pr = pr_df.head(100)

# Sentiment on titles
for title in sample_pr['title'].dropna():
    result = sentiment.analyze(title)
    print(f"{result['label']}: {title[:50]}...")

# Language on PR bodies
for body in sample_pr['body'].dropna():
    result = language.classify(body)
    print(f"{result['language_name']}: {body[:50]}...")

# Sentiment on comments
sample_comments = comments_df.head(100)
for comment in sample_comments['body'].dropna():
    result = sentiment.analyze(comment)
    print(f"{result['label']}: {comment[:50]}...")
```

## 🔄 Batch Processing Example

```python
import pandas as pd
from sentiment_analysis import SentimentAnalyzer
from language_classification import LanguageClassifier

# Load PR data
pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet")

# Initialize analyzers
sentiment = SentimentAnalyzer()
language = LanguageClassifier()

# Batch sentiment analysis on titles
titles = pr_df['title'].dropna().tolist()
sentiment_results = sentiment.analyze_batch(titles, batch_size=16)

# Batch language classification on bodies
bodies = pr_df['body'].dropna().head(1000).tolist()
language_results = language.classify_batch(bodies, batch_size=32)

# Add results back to dataframe
pr_df['title_sentiment'] = [r['label'] for r in sentiment_results]
pr_df['title_sentiment_confidence'] = [r['confidence'] for r in sentiment_results]
```

---

Made with ❤️ for PR Acceptability Analysis
