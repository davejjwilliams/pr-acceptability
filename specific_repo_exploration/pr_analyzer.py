#!/usr/bin/env python3
"""
PR Content Analyzer for AIDev Dataset

Analyzes pull request content using sentiment analysis and topic extraction.
Uses BERTopic for topic modeling and RoBERTa for sentiment analysis.

Usage:
    uv run specific_repo_exploration/pr_analyzer.py --repo "owner/repo"
    uv run specific_repo_exploration/pr_analyzer.py --repo "owner/repo" --sentiment-only
    uv run specific_repo_exploration/pr_analyzer.py --repo "owner/repo" --topics-only
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color scheme
OUTCOME_COLORS = {
    "accepted": "#4ECDC4",
    "rejected": "#FF6B6B",
}

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral": "#f39c12",
    "negative": "#e74c3c",
}


def load_datasets() -> dict:
    """Load required datasets from HuggingFace."""
    print("Loading datasets from HuggingFace...")
    datasets = {
        'pr_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet"),
        'repo_df': pd.read_parquet("hf://datasets/hao-li/AIDev/repository.parquet"),
        'pr_comments_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_comments.parquet"),
        'pr_reviews_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_reviews.parquet"),
        'pr_commit_details_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet"),
        'pr_timeline_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_timeline.parquet"),
    }
    return datasets


def annotate_pr_features(pr_df: pd.DataFrame, commit_details_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate PR dataframe with computed features."""
    df = pr_df.copy()

    df['turnaround_time'] = (
        pd.to_datetime(df['closed_at']) - pd.to_datetime(df['created_at'])
    ).dt.total_seconds()

    df['accepted'] = df['merged_at'].notna()
    df['rejected'] = (df['state'] == 'closed') & df['merged_at'].isna()

    return df


def identify_closed_not_planned(pr_df: pd.DataFrame, timeline_df: pd.DataFrame) -> pd.Series:
    """
    Identify PRs that were likely closed as 'not planned'.

    NOTE: GitHub's state_reason field is NOT in the AIDev dataset. This is a heuristic.
    """
    pr_ids = pr_df['id'].unique()
    not_planned_pr_ids = set()

    # Labels that indicate "not planned" closures
    not_planned_labels = [
        'duplicate', 'stale', 'wontfix', "won't fix", 'invalid', 'spam',
        'abandoned', 'superseded', 'not planned', 'wont-fix', 'not-planned'
    ]

    # Check labels (most reliable signal)
    if 'label' in timeline_df.columns:
        label_events = timeline_df[
            (timeline_df['pr_id'].isin(pr_ids)) &
            (timeline_df['event'] == 'labeled')
        ].copy()

        def is_not_planned_label(label):
            if pd.isna(label):
                return False
            label_lower = str(label).lower().strip()
            return any(kw in label_lower for kw in not_planned_labels)

        flagged_labels = label_events[label_events['label'].apply(is_not_planned_label)]
        not_planned_pr_ids.update(flagged_labels['pr_id'].unique())

    # Check close event messages
    close_events = timeline_df[
        (timeline_df['pr_id'].isin(pr_ids)) &
        (timeline_df['event'] == 'closed')
    ].copy()

    closure_phrases = [
        'closed as duplicate', 'closing as duplicate', 'marked as duplicate',
        'closed as stale', 'closing as stale', 'marked as stale',
        'closed as spam', 'closing as spam',
        'closed as not planned', 'closing as not planned',
        'superseded by', 'duplicate of #', 'duplicate of http'
    ]

    if 'message' in close_events.columns:
        def has_closure_phrase(text):
            if pd.isna(text):
                return False
            text_lower = str(text).lower()
            return any(phrase in text_lower for phrase in closure_phrases)

        close_events['is_not_planned'] = close_events['message'].apply(has_closure_phrase)
        not_planned_pr_ids.update(close_events[close_events['is_not_planned']]['pr_id'].unique())

    # Check PR title for explicit markers
    title_markers = ['[duplicate]', '[stale]', '[spam]', '[wontfix]', '[invalid]', '[superseded]']

    def check_title_markers(row):
        title = str(row.get('title', '')).lower()
        return any(marker in title for marker in title_markers)

    pr_title_flagged = pr_df.apply(check_title_markers, axis=1)

    is_not_planned = (
        pr_df['id'].isin(not_planned_pr_ids) |
        pr_title_flagged
    )

    return is_not_planned


def filter_pr_data(
    pr_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    min_turnaround_seconds: int = 60,
    exclude_not_planned: bool = True
) -> pd.DataFrame:
    """Apply filters to PR dataframe."""
    df = pr_df.copy()

    df = df[df['state'] == 'closed']
    df = df[df['turnaround_time'] >= min_turnaround_seconds]

    if exclude_not_planned:
        is_not_planned = identify_closed_not_planned(df, timeline_df)
        df = df[~is_not_planned]

    return df


def get_repo_prs(pr_df: pd.DataFrame, repo_df: pd.DataFrame, repo_identifier: str) -> tuple:
    """Get PRs for a specific repository."""
    repo_match = repo_df[repo_df['full_name'] == repo_identifier]

    if repo_match.empty:
        repo_match = repo_df[repo_df['full_name'].str.contains(repo_identifier, case=False, na=False)]

    if repo_match.empty:
        try:
            repo_id = int(repo_identifier)
            repo_match = repo_df[repo_df['id'] == repo_id]
        except ValueError:
            pass

    if repo_match.empty:
        raise ValueError(f"Repository '{repo_identifier}' not found")

    if len(repo_match) > 1:
        print(f"Multiple matches found:")
        for _, row in repo_match.iterrows():
            print(f"  - {row['full_name']}")
        raise ValueError("Please provide a more specific repository name")

    repo_info = repo_match.iloc[0].to_dict()
    repo_prs = pr_df[pr_df['repo_id'] == repo_info['id']]

    return repo_prs, repo_info


def aggregate_pr_texts(
    repo_pr_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    reviews_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate all text content for each PR."""
    pr_ids = repo_pr_df['id'].unique()

    results = []
    for pr_id in pr_ids:
        pr_row = repo_pr_df[repo_pr_df['id'] == pr_id].iloc[0]

        # PR title and body
        title = str(pr_row.get('title', '')) if pd.notna(pr_row.get('title')) else ''
        body = str(pr_row.get('body', '')) if pd.notna(pr_row.get('body')) else ''

        # Comments
        pr_comments = comments_df[comments_df['pr_id'] == pr_id]['body'].dropna().tolist()
        comments_text = ' '.join([str(c) for c in pr_comments])

        # Reviews
        pr_reviews = reviews_df[reviews_df['pr_id'] == pr_id]['body'].dropna().tolist()
        reviews_text = ' '.join([str(r) for r in pr_reviews])

        # Combined text
        all_text = f"{title} {body} {comments_text} {reviews_text}".strip()

        results.append({
            'pr_id': pr_id,
            'pr_number': pr_row.get('number'),
            'title': title,
            'body': body,
            'comments_text': comments_text,
            'reviews_text': reviews_text,
            'all_text': all_text,
            'accepted': pr_row.get('accepted'),
            'num_comments': len(pr_comments),
            'num_reviews': len(pr_reviews),
        })

    return pd.DataFrame(results)


class SentimentAnalyzer:
    """Sentiment analyzer using RoBERTa model."""

    MODEL_NAME = "siebert/sentiment-roberta-large-english"
    LABELS = ["negative", "positive"]  # This model uses NEGATIVE/POSITIVE labels

    def __init__(self, device: str = None):
        print(f"Loading sentiment model: {self.MODEL_NAME}...")

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, text: str) -> str:
        """Preprocess text for the model."""
        # Basic cleaning for technical text
        import re
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text

    def analyze(self, text: str) -> dict:
        """Analyze sentiment of a single text."""
        import torch
        from scipy.special import softmax

        if not text or len(text.strip()) == 0:
            return {
                "label": "neutral",
                "confidence": 1.0,
                "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
            }

        processed_text = self.preprocess(text[:512])  # Truncate for model
        
        if not processed_text or len(processed_text.strip()) == 0:
            return {
                "label": "neutral",
                "confidence": 1.0,
                "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
            }

        encoded_input = self.tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            output = self.model(**encoded_input)

        scores = output.logits[0].cpu().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)[::-1]
        
        # Map to include neutral based on confidence threshold
        label = self.LABELS[ranking[0]]
        confidence = float(scores[ranking[0]])
        
        # If confidence is low (< 0.6), consider it neutral
        if confidence < 0.6:
            final_label = "neutral"
        else:
            final_label = label

        return {
            "label": final_label,
            "confidence": confidence,
            "probabilities": {
                "negative": float(scores[0]),
                "neutral": 1.0 - confidence if confidence < 0.6 else 0.0,
                "positive": float(scores[1])
            }
        }

    def analyze_batch(self, texts: list, batch_size: int = 16) -> list:
        """Analyze sentiment of multiple texts."""
        import torch
        from scipy.special import softmax

        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Handle empty texts
            processed_batch = []
            valid_indices = []
            for j, t in enumerate(batch):
                if t and len(str(t).strip()) > 0:
                    processed_batch.append(self.preprocess(str(t)[:512]))
                    valid_indices.append(j)

            if not processed_batch:
                for _ in batch:
                    results.append({
                        "label": "neutral",
                        "confidence": 1.0,
                        "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
                    })
                continue
            
            # Filter out empty processed texts
            final_batch = []
            final_indices = []
            for idx, t in zip(valid_indices, processed_batch):
                if t and len(t.strip()) > 0:
                    final_batch.append(t)
                    final_indices.append(idx)
            
            if not final_batch:
                for _ in batch:
                    results.append({
                        "label": "neutral",
                        "confidence": 1.0,
                        "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
                    })
                continue
            
            processed_batch = final_batch
            valid_indices = final_indices

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

            # Map results back
            batch_results = []
            valid_idx = 0
            for j in range(len(batch)):
                if j in valid_indices:
                    probs = softmax(scores_batch[valid_idx])
                    ranking = np.argsort(probs)[::-1]
                    label = self.LABELS[ranking[0]]
                    confidence = float(probs[ranking[0]])
                    
                    # If confidence is low, consider it neutral
                    if confidence < 0.6:
                        final_label = "neutral"
                    else:
                        final_label = label
                    
                    batch_results.append({
                        "label": final_label,
                        "confidence": confidence,
                        "probabilities": {
                            "negative": float(probs[0]),
                            "neutral": 1.0 - confidence if confidence < 0.6 else 0.0,
                            "positive": float(probs[1])
                        }
                    })
                    valid_idx += 1
                else:
                    batch_results.append({
                        "label": "neutral",
                        "confidence": 1.0,
                        "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
                    })

            results.extend(batch_results)

        return results


def analyze_pr_sentiment(pr_texts_df: pd.DataFrame, max_prs: int = None) -> pd.DataFrame:
    """Analyze sentiment for all PRs."""
    print("Initializing sentiment analyzer...")
    analyzer = SentimentAnalyzer()

    df = pr_texts_df.copy()
    if max_prs and len(df) > max_prs:
        print(f"Sampling {max_prs} PRs from {len(df)} total")
        df = df.sample(n=max_prs, random_state=42)

    print(f"Analyzing sentiment for {len(df)} PRs...")

    # Analyze title sentiment
    print("  Analyzing titles...")
    title_sentiments = analyzer.analyze_batch(df['title'].tolist())
    df['title_sentiment'] = [s['label'] for s in title_sentiments]
    df['title_sentiment_confidence'] = [s['confidence'] for s in title_sentiments]

    # Analyze combined text sentiment
    print("  Analyzing full text...")
    all_sentiments = analyzer.analyze_batch(df['all_text'].tolist())
    df['overall_sentiment'] = [s['label'] for s in all_sentiments]
    df['overall_sentiment_confidence'] = [s['confidence'] for s in all_sentiments]
    df['sentiment_positive_prob'] = [s['probabilities']['positive'] for s in all_sentiments]
    df['sentiment_negative_prob'] = [s['probabilities']['negative'] for s in all_sentiments]
    df['sentiment_neutral_prob'] = [s['probabilities']['neutral'] for s in all_sentiments]

    return df


def extract_topics(pr_texts_df: pd.DataFrame, max_prs: int = None) -> tuple:
    """Extract topics from PR texts using BERTopic."""
    import re
    print("Initializing topic model...")

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    df = pr_texts_df.copy()
    if max_prs and len(df) > max_prs:
        print(f"Sampling {max_prs} PRs from {len(df)} total")
        df = df.sample(n=max_prs, random_state=42)

    # Filter out empty texts
    df = df[df['all_text'].str.len() > 10]

    if len(df) < 10:
        print("Not enough PRs with text content for topic modeling")
        return df, None, None

    # Preprocess texts to remove code and clean up
    def clean_text_for_topics(text):
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        # Remove inline code
        text = re.sub(r'`[^`]+`', '', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove file paths
        text = re.sub(r'[\w/]+\.[a-zA-Z]+', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    texts = [clean_text_for_topics(t) for t in df['all_text'].tolist()]
    print(f"Extracting topics from {len(texts)} PRs...")

    # Use sentence transformers for embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # CountVectorizer with stopwords removal for meaningful topic names
    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95  # Remove words that appear in >95% of documents
    )

    # BERTopic with vectorizer for better topic representation
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        min_topic_size=max(5, len(texts) // 20),  # Dynamic based on corpus size
        nr_topics="auto",
        verbose=True,
        calculate_probabilities=True
    )

    topics, probs = topic_model.fit_transform(texts)

    df['topic'] = topics
    df['topic_probability'] = [p.max() if hasattr(p, 'max') else (max(p) if isinstance(p, list) else p) for p in probs]

    # Get topic info
    topic_info = topic_model.get_topic_info()

    return df, topic_model, topic_info


def compute_sentiment_stats(sentiment_df: pd.DataFrame) -> dict:
    """Compute aggregated sentiment statistics."""
    total = len(sentiment_df)

    sentiment_counts = sentiment_df['overall_sentiment'].value_counts()

    stats = {
        'total_prs_analyzed': total,
        'positive_count': int(sentiment_counts.get('positive', 0)),
        'neutral_count': int(sentiment_counts.get('neutral', 0)),
        'negative_count': int(sentiment_counts.get('negative', 0)),
        'positive_rate': float(sentiment_counts.get('positive', 0) / total * 100) if total > 0 else 0,
        'neutral_rate': float(sentiment_counts.get('neutral', 0) / total * 100) if total > 0 else 0,
        'negative_rate': float(sentiment_counts.get('negative', 0) / total * 100) if total > 0 else 0,
        'avg_positive_prob': float(sentiment_df['sentiment_positive_prob'].mean()),
        'avg_negative_prob': float(sentiment_df['sentiment_negative_prob'].mean()),
        'avg_neutral_prob': float(sentiment_df['sentiment_neutral_prob'].mean()),
    }

    # By outcome
    for outcome in ['accepted', 'rejected']:
        subset = sentiment_df[sentiment_df['accepted'] == (outcome == 'accepted')]
        if len(subset) > 0:
            outcome_counts = subset['overall_sentiment'].value_counts()
            stats[f'{outcome}_positive_rate'] = float(outcome_counts.get('positive', 0) / len(subset) * 100)
            stats[f'{outcome}_negative_rate'] = float(outcome_counts.get('negative', 0) / len(subset) * 100)
            stats[f'{outcome}_neutral_rate'] = float(outcome_counts.get('neutral', 0) / len(subset) * 100)

    return stats


# ============ Visualization Functions ============

def create_sentiment_distribution(sentiment_df: pd.DataFrame, output_dir: Path) -> None:
    """Create sentiment distribution visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall distribution
    ax = axes[0]
    sentiment_counts = sentiment_df['overall_sentiment'].value_counts()
    colors = [SENTIMENT_COLORS[s] for s in sentiment_counts.index]

    ax.pie(sentiment_counts.values, labels=sentiment_counts.index, colors=colors,
           autopct='%1.1f%%', startangle=90)
    ax.set_title('Overall Sentiment Distribution')

    # Bar chart
    ax = axes[1]
    x = np.arange(len(sentiment_counts))
    bars = ax.bar(x, sentiment_counts.values, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of PRs')
    ax.set_title('Sentiment Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(sentiment_counts.index)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, sentiment_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sentiment_distribution.pdf', bbox_inches='tight')
    plt.close()


def create_sentiment_by_outcome(sentiment_df: pd.DataFrame, output_dir: Path) -> None:
    """Create sentiment by outcome visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Grouped bar chart
    ax = axes[0]

    outcomes = ['Accepted', 'Rejected']
    sentiments = ['positive', 'neutral', 'negative']

    x = np.arange(len(outcomes))
    width = 0.25

    accepted_counts = sentiment_df[sentiment_df['accepted'] == True]['overall_sentiment'].value_counts()
    rejected_counts = sentiment_df[sentiment_df['accepted'] == False]['overall_sentiment'].value_counts()

    for i, sentiment in enumerate(sentiments):
        counts = [
            accepted_counts.get(sentiment, 0),
            rejected_counts.get(sentiment, 0)
        ]
        ax.bar(x + i * width, counts, width, label=sentiment.capitalize(),
               color=SENTIMENT_COLORS[sentiment], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Outcome')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment by PR Outcome')
    ax.set_xticks(x + width)
    ax.set_xticklabels(outcomes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Stacked percentage bar chart
    ax = axes[1]

    accepted_total = len(sentiment_df[sentiment_df['accepted'] == True])
    rejected_total = len(sentiment_df[sentiment_df['accepted'] == False])

    accepted_pcts = [accepted_counts.get(s, 0) / accepted_total * 100 if accepted_total > 0 else 0 for s in sentiments]
    rejected_pcts = [rejected_counts.get(s, 0) / rejected_total * 100 if rejected_total > 0 else 0 for s in sentiments]

    bottom_accepted = 0
    bottom_rejected = 0

    for sentiment, acc_pct, rej_pct in zip(sentiments, accepted_pcts, rejected_pcts):
        ax.barh(['Accepted'], [acc_pct], left=[bottom_accepted],
                color=SENTIMENT_COLORS[sentiment], edgecolor='black', linewidth=0.5, label=sentiment.capitalize())
        ax.barh(['Rejected'], [rej_pct], left=[bottom_rejected],
                color=SENTIMENT_COLORS[sentiment], edgecolor='black', linewidth=0.5)
        bottom_accepted += acc_pct
        bottom_rejected += rej_pct

    ax.set_xlabel('Percentage')
    ax.set_title('Sentiment Distribution by Outcome')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'sentiment_by_outcome.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sentiment_by_outcome.pdf', bbox_inches='tight')
    plt.close()


def create_topic_distribution(topic_df: pd.DataFrame, topic_info: pd.DataFrame, output_dir: Path) -> None:
    """Create topic distribution visualization."""
    if topic_info is None or topic_info.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Top 15 topics (excluding -1 which is outliers)
    top_topics = topic_info[topic_info['Topic'] != -1].head(15)

    if top_topics.empty:
        plt.close()
        return

    y = np.arange(len(top_topics))
    bars = ax.barh(y, top_topics['Count'].values, color='#3498db', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Number of PRs')
    ax.set_ylabel('Topic')
    ax.set_title('Top Topics in PR Discussions')
    ax.set_yticks(y)

    # Create topic labels from keywords
    labels = []
    for _, row in top_topics.iterrows():
        topic_num = row['Topic']
        name = row.get('Name', f'Topic {topic_num}')
        # Truncate long names
        if len(str(name)) > 40:
            name = str(name)[:40] + '...'
        labels.append(name)

    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'topic_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'topic_distribution.pdf', bbox_inches='tight')
    plt.close()


def generate_content_report(
    repo_info: dict,
    sentiment_stats: dict,
    sentiment_df: pd.DataFrame,
    topic_info: pd.DataFrame = None,
    output_dir: Path = None
) -> str:
    """Generate markdown report for content analysis."""
    report = []
    report.append(f"# PR Content Analysis Report: {repo_info.get('full_name', 'Unknown')}")
    report.append("")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Sentiment Analysis
    report.append("## Sentiment Analysis")
    report.append("")
    report.append(f"**Total PRs Analyzed:** {sentiment_stats['total_prs_analyzed']}")
    report.append("")
    report.append("### Overall Distribution")
    report.append("")
    report.append(f"- Positive: {sentiment_stats['positive_count']} ({sentiment_stats['positive_rate']:.1f}%)")
    report.append(f"- Neutral: {sentiment_stats['neutral_count']} ({sentiment_stats['neutral_rate']:.1f}%)")
    report.append(f"- Negative: {sentiment_stats['negative_count']} ({sentiment_stats['negative_rate']:.1f}%)")
    report.append("")

    report.append("### By Outcome")
    report.append("")
    report.append("| Outcome | Positive | Neutral | Negative |")
    report.append("|---------|----------|---------|----------|")

    for outcome in ['accepted', 'rejected']:
        pos = sentiment_stats.get(f'{outcome}_positive_rate', 0)
        neu = sentiment_stats.get(f'{outcome}_neutral_rate', 0)
        neg = sentiment_stats.get(f'{outcome}_negative_rate', 0)
        report.append(f"| {outcome.capitalize()} | {pos:.1f}% | {neu:.1f}% | {neg:.1f}% |")
    report.append("")

    # Topic Analysis
    if topic_info is not None and not topic_info.empty:
        report.append("## Topic Analysis")
        report.append("")

        # Top topics
        top_topics = topic_info[topic_info['Topic'] != -1].head(10)
        if not top_topics.empty:
            report.append("### Top Topics")
            report.append("")
            report.append("| Topic | Name | Count |")
            report.append("|-------|------|-------|")
            for _, row in top_topics.iterrows():
                name = str(row.get('Name', f"Topic {row['Topic']}"))[:50]
                report.append(f"| {row['Topic']} | {name} | {row['Count']} |")
            report.append("")

        # Outliers
        outliers = topic_info[topic_info['Topic'] == -1]
        if not outliers.empty:
            outlier_count = outliers['Count'].values[0]
            report.append(f"**Outlier PRs (no clear topic):** {outlier_count}")
            report.append("")

    return "\n".join(report)


def analyze_pr_content(
    repo_identifier: str,
    output_base_dir: str = "./specific_repo_exploration",
    min_turnaround: int = 60,
    include_not_planned: bool = False,
    max_prs: int = None,
    sentiment_only: bool = False,
    topics_only: bool = False
) -> None:
    """Main function to analyze PR content."""

    # Load datasets
    datasets = load_datasets()

    # Annotate PR features
    print("Annotating PR features...")
    pr_df = annotate_pr_features(datasets['pr_df'], datasets['pr_commit_details_df'])

    # Get repository PRs
    print(f"Finding repository: {repo_identifier}")
    repo_pr_df, repo_info = get_repo_prs(pr_df, datasets['repo_df'], repo_identifier)
    print(f"Found {len(repo_pr_df)} PRs for {repo_info['full_name']}")

    # Apply filters
    print("Applying filters...")
    repo_pr_df = filter_pr_data(
        repo_pr_df,
        datasets['pr_timeline_df'],
        min_turnaround_seconds=min_turnaround,
        exclude_not_planned=not include_not_planned
    )
    print(f"After filtering: {len(repo_pr_df)} PRs")

    if len(repo_pr_df) == 0:
        print("No PRs remaining after filtering. Exiting.")
        return

    # Create output directories
    repo_name_safe = repo_info['full_name'].replace('/', '_')
    output_dir = Path(output_base_dir) / repo_name_safe
    sentiment_dir = output_dir / 'sentiment_analysis'
    topic_dir = output_dir / 'topic_analysis'

    output_dir.mkdir(parents=True, exist_ok=True)
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    topic_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate PR texts
    print("Aggregating PR texts...")
    pr_texts_df = aggregate_pr_texts(
        repo_pr_df,
        datasets['pr_comments_df'],
        datasets['pr_reviews_df']
    )

    sentiment_df = None
    sentiment_stats = None
    topic_df = None
    topic_model = None
    topic_info = None

    # Sentiment Analysis
    if not topics_only:
        print("\n=== Sentiment Analysis ===")
        sentiment_df = analyze_pr_sentiment(pr_texts_df, max_prs)
        sentiment_stats = compute_sentiment_stats(sentiment_df)

        # Save results
        sentiment_df.to_csv(sentiment_dir / 'pr_sentiments.csv', index=False)
        with open(sentiment_dir / 'sentiment_stats.json', 'w') as f:
            json.dump(sentiment_stats, f, indent=2)

        # Visualizations
        print("Creating sentiment visualizations...")
        create_sentiment_distribution(sentiment_df, sentiment_dir)
        create_sentiment_by_outcome(sentiment_df, sentiment_dir)

    # Topic Analysis
    if not sentiment_only:
        print("\n=== Topic Analysis ===")
        topic_df, topic_model, topic_info = extract_topics(pr_texts_df, max_prs)

        if topic_df is not None:
            # Save results
            topic_df.to_csv(topic_dir / 'pr_topics.csv', index=False)
            if topic_info is not None:
                topic_info.to_csv(topic_dir / 'topic_info.csv', index=False)

            # Visualizations
            print("Creating topic visualizations...")
            create_topic_distribution(topic_df, topic_info, topic_dir)

    # Generate report
    print("\nGenerating report...")
    if sentiment_stats is None:
        sentiment_stats = {'total_prs_analyzed': 0, 'positive_count': 0, 'neutral_count': 0, 'negative_count': 0,
                          'positive_rate': 0, 'neutral_rate': 0, 'negative_rate': 0}
    if sentiment_df is None:
        sentiment_df = pd.DataFrame()

    report = generate_content_report(
        repo_info, sentiment_stats, sentiment_df, topic_info, output_dir
    )
    with open(output_dir / 'content_analysis_report.md', 'w') as f:
        f.write(report)

    print(f"\nContent analysis complete! Results saved to: {output_dir}")
    print(f"  - sentiment_analysis/")
    print(f"  - topic_analysis/")
    print(f"  - content_analysis_report.md")


def main():
    parser = argparse.ArgumentParser(
        description="PR content analysis (sentiment and topics) for AIDev dataset"
    )

    # Repository selection
    repo_group = parser.add_mutually_exclusive_group(required=True)
    repo_group.add_argument("--repo", "-r", type=str,
                           help="Repository full name (e.g., 'owner/repo')")
    repo_group.add_argument("--repo-id", type=int,
                           help="Repository ID")

    # Output options
    parser.add_argument("--output", "-o", type=str,
                       default="./specific_repo_exploration",
                       help="Output base directory")

    # Filtering options
    parser.add_argument("--min-turnaround", type=int, default=60,
                       help="Minimum turnaround time in seconds (default: 60)")
    parser.add_argument("--include-not-planned", action="store_true",
                       help="Include PRs that were 'closed as not planned'")

    # Analysis options
    parser.add_argument("--max-prs", type=int, default=None,
                       help="Maximum number of PRs to analyze (for large repos)")
    parser.add_argument("--sentiment-only", action="store_true",
                       help="Only run sentiment analysis")
    parser.add_argument("--topics-only", action="store_true",
                       help="Only run topic analysis")

    args = parser.parse_args()

    repo_identifier = args.repo if args.repo else str(args.repo_id)

    analyze_pr_content(
        repo_identifier=repo_identifier,
        output_base_dir=args.output,
        min_turnaround=args.min_turnaround,
        include_not_planned=args.include_not_planned,
        max_prs=args.max_prs,
        sentiment_only=args.sentiment_only,
        topics_only=args.topics_only
    )


if __name__ == "__main__":
    main()
