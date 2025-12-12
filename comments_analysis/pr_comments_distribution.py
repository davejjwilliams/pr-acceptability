#!/usr/bin/env python3
"""
PR Comments Distribution Analysis

Analyzes the distribution of number of comments for accepted PRs (merged)
vs rejected PRs (closed but not merged).

Usage:
    uv run comments_analysis/pr_comments_distribution.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

    # Filter repositories by stars (repository table uses 'id' column)
    high_star_repos = repo_df[repo_df['stars'] > min_stars]['id'].unique()
    print(f"Found {len(high_star_repos)} repositories with > {min_stars} stars")

    # Filter PRs from high-star repositories (PR table uses 'repo_id')
    filtered_pr_df = pr_df[pr_df['repo_id'].isin(high_star_repos)].copy()
    print(f"Filtered to {len(filtered_pr_df)} PRs from high-star repositories")

    return filtered_pr_df


def count_comments_per_pr(pr_df, comments_df):
    """Count number of comments for each PR."""
    print("\nCounting comments per PR...")

    # Count comments per PR
    comments_count = comments_df.groupby('pr_id').size().reset_index(name='num_comments')

    # Merge with PR data
    pr_with_comments = pr_df.merge(comments_count, left_on='id', right_on='pr_id', how='left')

    # Fill PRs with no comments with 0
    pr_with_comments['num_comments'] = pr_with_comments['num_comments'].fillna(0).astype(int)

    print(f"PRs with comment counts: {len(pr_with_comments)}")

    return pr_with_comments


def categorize_prs(pr_df):
    """Categorize PRs as accepted (merged) or rejected (closed but not merged)."""
    print("\nCategorizing PRs...")

    # Accepted PRs: merged_at is not null
    accepted_mask = pr_df['merged_at'].notna()

    # Rejected PRs: closed_at is not null AND merged_at is null
    rejected_mask = (pr_df['closed_at'].notna()) & (pr_df['merged_at'].isna())

    accepted_prs = pr_df[accepted_mask].copy()
    rejected_prs = pr_df[rejected_mask].copy()

    print(f"Accepted PRs (merged): {len(accepted_prs)}")
    print(f"Rejected PRs (closed, not merged): {len(rejected_prs)}")

    # Print statistics
    print(f"\nComment statistics for accepted PRs:")
    print(f"  Mean: {accepted_prs['num_comments'].mean():.2f}")
    print(f"  Median: {accepted_prs['num_comments'].median():.0f}")
    print(f"  Max: {accepted_prs['num_comments'].max():.0f}")

    print(f"\nComment statistics for rejected PRs:")
    print(f"  Mean: {rejected_prs['num_comments'].mean():.2f}")
    print(f"  Median: {rejected_prs['num_comments'].median():.0f}")
    print(f"  Max: {rejected_prs['num_comments'].max():.0f}")

    return accepted_prs, rejected_prs


def create_histograms(accepted_prs, rejected_prs, max_comments=50):
    """Create histograms showing distribution of comments for accepted vs rejected PRs."""
    print(f"\nCreating histograms (showing up to {max_comments} comments)...")

    # Filter to reasonable range for better visualization
    accepted_filtered = accepted_prs[accepted_prs['num_comments'] <= max_comments]
    rejected_filtered = rejected_prs[rejected_prs['num_comments'] <= max_comments]

    print(f"Showing {len(accepted_filtered)}/{len(accepted_prs)} accepted PRs")
    print(f"Showing {len(rejected_filtered)}/{len(rejected_prs)} rejected PRs")

    # Set up the plot style
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Define bins
    bins = np.arange(0, max_comments + 2, 1)

    # Histogram for accepted PRs (as percentage)
    weights_accepted = np.ones_like(accepted_filtered['num_comments']) / len(accepted_prs) * 100
    ax1.hist(accepted_filtered['num_comments'], bins=bins, weights=weights_accepted,
             color='#56B4E9', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Number of Comments', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Percentage of PRs (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Comments for Accepted PRs (Merged)\n(Repositories with >100 stars)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add statistics text
    stats_text = (f"Total: {len(accepted_prs)} PRs\n"
                  f"Mean: {accepted_prs['num_comments'].mean():.1f}\n"
                  f"Median: {accepted_prs['num_comments'].median():.0f}")
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Histogram for rejected PRs (as percentage)
    weights_rejected = np.ones_like(rejected_filtered['num_comments']) / len(rejected_prs) * 100
    ax2.hist(rejected_filtered['num_comments'], bins=bins, weights=weights_rejected,
             color='#D55E00', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Number of Comments', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Percentage of PRs (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Comments for Rejected PRs (Closed, Not Merged)\n(Repositories with >100 stars)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Add statistics text
    stats_text = (f"Total: {len(rejected_prs)} PRs\n"
                  f"Mean: {rejected_prs['num_comments'].mean():.1f}\n"
                  f"Median: {rejected_prs['num_comments'].median():.0f}")
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_path = 'comments_analysis/pr_comments_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Also save as PDF
    pdf_path = 'comments_analysis/pr_comments_distribution.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")

    plt.show()


def save_statistics(accepted_prs, rejected_prs):
    """Save detailed statistics to CSV."""
    print("\nSaving statistics...")

    # Create summary statistics
    stats = []

    for label, df in [('Accepted', accepted_prs), ('Rejected', rejected_prs)]:
        stats.append({
            'category': label,
            'total_prs': len(df),
            'mean_comments': df['num_comments'].mean(),
            'median_comments': df['num_comments'].median(),
            'std_comments': df['num_comments'].std(),
            'min_comments': df['num_comments'].min(),
            'max_comments': df['num_comments'].max(),
            'q25_comments': df['num_comments'].quantile(0.25),
            'q75_comments': df['num_comments'].quantile(0.75)
        })

    stats_df = pd.DataFrame(stats)

    output_path = 'comments_analysis/pr_comments_stats.csv'
    stats_df.to_csv(output_path, index=False)
    print(f"Statistics saved to: {output_path}")

    return stats_df


def main():
    """Main execution function."""
    print("="*60)
    print("PR Comments Distribution Analysis")
    print("Comparing accepted vs rejected PRs")
    print("(Repositories with >100 stars)")
    print("="*60)

    # Load datasets
    pr_df, repo_df, comments_df = load_datasets()

    # Filter to high-star repositories
    pr_df = filter_high_star_repos(pr_df, repo_df, min_stars=100)

    # Count comments per PR
    pr_with_comments = count_comments_per_pr(pr_df, comments_df)

    # Categorize PRs
    accepted_prs, rejected_prs = categorize_prs(pr_with_comments)

    # Create visualizations
    create_histograms(accepted_prs, rejected_prs, max_comments=50)

    # Save statistics
    stats_df = save_statistics(accepted_prs, rejected_prs)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
