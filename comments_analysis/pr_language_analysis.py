#!/usr/bin/env python3
"""
PR Language Analysis Script

Analyzes the relationship between PR title language and successful PR closures
for repositories with more than 100 stars.

Usage:
    uv run comments_analysis/pr_language_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from language_classification import LanguageClassifier
import warnings

warnings.filterwarnings('ignore')


def load_datasets():
    """Load pull request and repository datasets."""
    print("Loading datasets...")

    pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet")
    repo_df = pd.read_parquet("hf://datasets/hao-li/AIDev/repository.parquet")

    print(f"Loaded {len(pr_df)} pull requests")
    print(f"Loaded {len(repo_df)} repositories")

    return pr_df, repo_df


def filter_high_star_repos(pr_df, repo_df, min_stars=100):
    """Filter PRs from repositories with more than min_stars stars."""
    print(f"\nFiltering repositories with > {min_stars} stars...")

    # Filter repositories by stars (repository table uses 'id' column)
    high_star_repos = repo_df[repo_df['stars'] > min_stars]['id'].unique()
    print(f"Found {len(high_star_repos)} repositories with > {min_stars} stars")

    # Filter PRs from high-star repositories (PR table uses 'repo_id')
    filtered_pr_df = pr_df[pr_df['repo_id'].isin(high_star_repos)].copy()
    print(f"Filtered to {len(filtered_pr_df)} PRs from high-star repositories")

    return filtered_pr_df


def classify_pr_languages(pr_df):
    """Classify the language of PR titles."""
    print("\nInitializing language classifier...")
    classifier = LanguageClassifier()

    print("Classifying PR title languages...")

    # Filter out PRs with missing titles
    pr_df_with_titles = pr_df[pr_df['title'].notna()].copy()
    print(f"Processing {len(pr_df_with_titles)} PRs with titles")

    # Classify in batches for efficiency
    batch_size = 1000
    all_results = []

    for i in range(0, len(pr_df_with_titles), batch_size):
        batch = pr_df_with_titles.iloc[i:i+batch_size]
        titles = batch['title'].tolist()

        results = classifier.classify_batch(titles, top_k=1)
        all_results.extend(results)

        if (i + batch_size) % 5000 == 0:
            print(f"  Processed {min(i + batch_size, len(pr_df_with_titles))}/{len(pr_df_with_titles)} PRs")

    print(f"Completed classification of {len(all_results)} PRs")

    # Add language information to dataframe
    pr_df_with_titles['language_code'] = [r['language_code'] for r in all_results]
    pr_df_with_titles['language_name'] = [r['language_name'] for r in all_results]
    pr_df_with_titles['language_confidence'] = [r['confidence'] for r in all_results]

    return pr_df_with_titles


def analyze_successful_closures(pr_df):
    """Analyze successful PR closures by language."""
    print("\nAnalyzing successful closures by language...")

    # Determine successful closures (merged PRs have non-null merged_at)
    pr_df['successfully_closed'] = pr_df['merged_at'].notna()

    # Group by language and count successful closures
    language_stats = pr_df.groupby('language_name').agg({
        'successfully_closed': ['sum', 'count']
    }).reset_index()

    language_stats.columns = ['language_name', 'successful_count', 'total_count']
    language_stats['success_rate'] = (language_stats['successful_count'] /
                                      language_stats['total_count'] * 100)

    # Sort by total count descending
    language_stats = language_stats.sort_values('total_count', ascending=False)

    print(f"\nLanguage statistics:")
    print(language_stats.head(10))

    return language_stats


def create_plot(language_stats, top_n=15):
    """Create a bar plot of successful closure rates by language."""
    print(f"\nCreating plot for top {top_n} languages...")

    # Take top N languages by total count
    top_languages = language_stats.head(top_n)

    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))

    # Create bar plot with success rate (percentage)
    ax = plt.subplot(111)
    x_pos = range(len(top_languages))

    bars = ax.bar(x_pos, top_languages['success_rate'],
                   color='#56B4E9', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Customize plot
    ax.set_xlabel('Language', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('PR Merge Success Rate by PR Title Language\n(Repositories with >100 stars)',
                 fontsize=14, fontweight='bold', pad=20)

    # Set y-axis limits
    ax.set_ylim(0, 100)

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_languages['language_name'], rotation=45, ha='right')

    # Add value labels on top of bars with sample size
    for i, (idx, row) in enumerate(top_languages.iterrows()):
        height = row['success_rate']
        ax.text(i, height, f"{height:.1f}%\n(n={int(row['total_count'])})",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_path = 'comments_analysis/pr_language_closures.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Also save as PDF
    pdf_path = 'comments_analysis/pr_language_closures.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")

    plt.show()


def save_results(language_stats):
    """Save analysis results to CSV."""
    output_path = 'comments_analysis/pr_language_stats.csv'
    language_stats.to_csv(output_path, index=False)
    print(f"\nLanguage statistics saved to: {output_path}")


def main():
    """Main execution function."""
    print("="*60)
    print("PR Language Analysis")
    print("Analyzing PR title languages and successful closures")
    print("="*60)

    # Load datasets
    pr_df, repo_df = load_datasets()

    # Filter to high-star repositories
    filtered_pr_df = filter_high_star_repos(pr_df, repo_df, min_stars=100)

    # Classify PR title languages
    pr_with_languages = classify_pr_languages(filtered_pr_df)

    # Analyze successful closures by language
    language_stats = analyze_successful_closures(pr_with_languages)

    # Create visualization
    create_plot(language_stats, top_n=15)

    # Save results
    save_results(language_stats)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
