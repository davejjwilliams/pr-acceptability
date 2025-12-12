#!/usr/bin/env python3
"""
PR Language and Comments Analysis

Analyzes the relationship between PR title language, number of comments,
and merge probability for repositories with more than 100 stars.

Creates:
1. Distribution of comments by language
2. Correlation between language, number of comments, and merge probability

Usage:
    uv run comments_analysis/pr_language_comments_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from language_classification import LanguageClassifier
import warnings

warnings.filterwarnings('ignore')


def load_datasets():
    """Load pull request, repository, and comments datasets."""
    print("Loading datasets...")

    pr_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet")
    repo_df = pd.read_parquet("hf://datasets/hao-li/AIDev/repository.parquet")
    comments_df = pd.read_parquet("hf://datasets/hao-li/AIDev/pr_comments.parquet")

    print(f"Loaded {len(pr_df)} pull requests")
    print(f"Loaded {len(repo_df)} repositories")
    print(f"Loaded {len(comments_df)} PR comments")

    return pr_df, repo_df, comments_df


def filter_high_star_repos(pr_df, repo_df, min_stars=100):
    """Filter PRs from repositories with more than min_stars stars."""
    print(f"\nFiltering repositories with > {min_stars} stars...")

    high_star_repos = repo_df[repo_df['stars'] > min_stars]['id'].unique()
    print(f"Found {len(high_star_repos)} repositories with > {min_stars} stars")

    filtered_pr_df = pr_df[pr_df['repo_id'].isin(high_star_repos)].copy()
    print(f"Filtered to {len(filtered_pr_df)} PRs from high-star repositories")

    return filtered_pr_df


def classify_pr_languages(pr_df):
    """Classify the language of PR titles."""
    print("\nInitializing language classifier...")
    classifier = LanguageClassifier()

    print("Classifying PR title languages...")

    pr_df_with_titles = pr_df[pr_df['title'].notna()].copy()
    print(f"Processing {len(pr_df_with_titles)} PRs with titles")

    # Classify in batches
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

    pr_df_with_titles['language_code'] = [r['language_code'] for r in all_results]
    pr_df_with_titles['language_name'] = [r['language_name'] for r in all_results]
    pr_df_with_titles['language_confidence'] = [r['confidence'] for r in all_results]

    return pr_df_with_titles


def count_comments_per_pr(pr_df, comments_df):
    """Count number of comments for each PR."""
    print("\nCounting comments per PR...")

    comments_count = comments_df.groupby('pr_id').size().reset_index(name='num_comments')
    pr_with_comments = pr_df.merge(comments_count, left_on='id', right_on='pr_id', how='left')
    pr_with_comments['num_comments'] = pr_with_comments['num_comments'].fillna(0).astype(int)

    print(f"PRs with comment counts: {len(pr_with_comments)}")

    return pr_with_comments


def analyze_language_comments(pr_df):
    """Analyze comments distribution by language and merge probability."""
    print("\nAnalyzing language-comments relationship...")

    # Add merge status
    pr_df['is_merged'] = pr_df['merged_at'].notna()

    # Group by language
    language_stats = pr_df.groupby('language_name').agg({
        'num_comments': ['mean', 'median', 'std', 'count'],
        'is_merged': ['sum', 'mean']
    }).reset_index()

    language_stats.columns = ['language_name', 'mean_comments', 'median_comments',
                               'std_comments', 'total_prs', 'merged_prs', 'merge_rate']

    # Convert merge rate to percentage
    language_stats['merge_rate'] = language_stats['merge_rate'] * 100

    # Filter languages with at least 10 PRs for statistical significance
    language_stats = language_stats[language_stats['total_prs'] >= 10].copy()

    # Sort by total PRs
    language_stats = language_stats.sort_values('total_prs', ascending=False)

    print(f"\nFound {len(language_stats)} languages with >= 10 PRs")
    print(f"\nTop 10 languages by PR count:")
    print(language_stats.head(10)[['language_name', 'total_prs', 'mean_comments', 'merge_rate']])

    return language_stats, pr_df


def create_comments_distribution_plot(pr_df, top_n=15):
    """Create box plot showing distribution of comments by language."""
    print(f"\nCreating comments distribution plot for top {top_n} languages...")

    # Get top languages by PR count
    top_languages = pr_df.groupby('language_name').size().sort_values(ascending=False).head(top_n).index

    # Filter data
    plot_data = pr_df[pr_df['language_name'].isin(top_languages)].copy()

    # Cap comments at 20 for better visualization
    plot_data['num_comments_capped'] = plot_data['num_comments'].clip(upper=20)

    # Set up plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 8))

    # Create box plot
    ax = plt.subplot(111)

    # Order by total PR count
    language_order = plot_data.groupby('language_name').size().sort_values(ascending=False).index

    sns.boxplot(data=plot_data, x='language_name', y='num_comments_capped',
                order=language_order, color='#56B4E9', ax=ax)

    # Customize plot
    ax.set_xlabel('Language', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Comments (capped at 20)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of PR Comments by Language\n(Repositories with >100 stars)',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add sample size annotations
    for i, lang in enumerate(language_order):
        count = len(plot_data[plot_data['language_name'] == lang])
        median = plot_data[plot_data['language_name'] == lang]['num_comments_capped'].median()
        ax.text(i, -1.5, f'n={count}', ha='center', va='top', fontsize=8, color='gray')

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save plot
    output_path = 'comments_analysis/pr_comments_by_language.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    pdf_path = 'comments_analysis/pr_comments_by_language.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")

    plt.close()


def create_correlation_plot(language_stats, top_n=20):
    """Create plot showing correlation between language, comments, and merge rate."""
    print(f"\nCreating correlation plot for top {top_n} languages...")

    # Get top languages
    top_stats = language_stats.head(top_n).copy()

    # Set up plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: Dual-axis plot (merge rate and mean comments)
    x_pos = range(len(top_stats))

    # Bar plot for merge rate
    bars = ax1.bar(x_pos, top_stats['merge_rate'], color='#56B4E9', alpha=0.7,
                   edgecolor='black', linewidth=0.5, label='Merge Rate (%)')

    ax1.set_xlabel('Language', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Merge Rate (%)', fontsize=11, fontweight='bold', color='#56B4E9')
    ax1.tick_params(axis='y', labelcolor='#56B4E9')
    ax1.set_ylim(0, 100)

    # Create second y-axis for mean comments
    ax1_twin = ax1.twinx()
    line = ax1_twin.plot(x_pos, top_stats['mean_comments'], color='#D55E00',
                         marker='o', linewidth=2, markersize=8, label='Mean Comments')
    ax1_twin.set_ylabel('Mean Number of Comments', fontsize=11, fontweight='bold', color='#D55E00')
    ax1_twin.tick_params(axis='y', labelcolor='#D55E00')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(top_stats['language_name'], rotation=45, ha='right')
    ax1.set_title('Merge Rate vs Mean Comments by Language\n(Repositories with >100 stars)',
                  fontsize=12, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Plot 2: Scatter plot (mean comments vs merge rate)
    # Size represents number of PRs
    sizes = (top_stats['total_prs'] / top_stats['total_prs'].max()) * 1000

    scatter = ax2.scatter(top_stats['mean_comments'], top_stats['merge_rate'],
                         s=sizes, alpha=0.6, c=range(len(top_stats)),
                         cmap='viridis', edgecolors='black', linewidth=1)

    # Add language labels
    for idx, row in top_stats.iterrows():
        ax2.annotate(row['language_name'],
                    (row['mean_comments'], row['merge_rate']),
                    fontsize=8, alpha=0.7, ha='center', va='bottom')

    ax2.set_xlabel('Mean Number of Comments', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Merge Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Correlation: Comments vs Merge Rate\n(Bubble size = number of PRs)',
                  fontsize=12, fontweight='bold', pad=15)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 100)

    plt.tight_layout()

    # Save plot
    output_path = 'comments_analysis/pr_language_comments_correlation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    pdf_path = 'comments_analysis/pr_language_comments_correlation.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")

    plt.close()


def save_results(language_stats):
    """Save analysis results to CSV."""
    output_path = 'comments_analysis/pr_language_comments_stats.csv'
    language_stats.to_csv(output_path, index=False)
    print(f"\nLanguage-comments statistics saved to: {output_path}")


def main():
    """Main execution function."""
    print("="*70)
    print("PR Language and Comments Analysis")
    print("Analyzing relationship between language, comments, and merge rate")
    print("="*70)

    # Load datasets
    pr_df, repo_df, comments_df = load_datasets()

    # Filter to high-star repositories
    filtered_pr_df = filter_high_star_repos(pr_df, repo_df, min_stars=100)

    # Count comments per PR
    pr_with_comments = count_comments_per_pr(filtered_pr_df, comments_df)

    # Classify PR title languages
    pr_with_languages = classify_pr_languages(pr_with_comments)

    # Analyze language-comments relationship
    language_stats, pr_data = analyze_language_comments(pr_with_languages)

    # Create visualizations
    create_comments_distribution_plot(pr_data, top_n=15)
    create_correlation_plot(language_stats, top_n=20)

    # Save results
    save_results(language_stats)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
