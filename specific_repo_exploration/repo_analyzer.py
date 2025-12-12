#!/usr/bin/env python3
"""
Repository Analyzer for AIDev Dataset

Analyzes pull request statistics for a specific repository in the AIDev dataset.
Generates comprehensive statistics, visualizations, and reports.

Usage:
    uv run specific_repo_exploration/repo_analyzer.py --repo "owner/repo"
    uv run specific_repo_exploration/repo_analyzer.py --repo-id 123456
    uv run specific_repo_exploration/repo_analyzer.py --list-repos
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from cliffs_delta import cliffs_delta
except ImportError:
    cliffs_delta = None

from utils import filter_top_n_for_cols

# Color scheme from helper.py
COLOR_MAP = {
    "Human": "#56B4E9",
    "OpenAI_Codex": "#D55E00",
    "OpenAI Codex": "#D55E00",
    "Devin": "#009E73",
    "Copilot": "#0072B2",
    "GitHub Copilot": "#0072B2",
    "Cursor": "#785EF0",
    "Claude_Code": "#DC267F",
    "Claude Code": "#DC267F",
}

OUTCOME_COLORS = {
    "accepted": "#4ECDC4",
    "rejected": "#FF6B6B",
}

TASK_TYPE_ORDER = [
    "feat", "fix", "perf", "refactor", "style",
    "docs", "test", "chore", "build", "ci", "revert", "other"
]


def load_datasets() -> dict:
    """Load all required datasets from HuggingFace."""
    print("Loading datasets from HuggingFace...")
    datasets = {
        'pr_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet"),
        'repo_df': pd.read_parquet("hf://datasets/hao-li/AIDev/repository.parquet"),
        'pr_comments_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_comments.parquet"),
        'pr_reviews_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_reviews.parquet"),
        'pr_review_comments_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_review_comments_v2.parquet"),
        'pr_commits_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commits.parquet"),
        'pr_commit_details_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet"),
        'pr_timeline_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_timeline.parquet"),
        'pr_task_type_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_task_type.parquet"),
    }
    print(f"Loaded {len(datasets['pr_df'])} PRs from {len(datasets['repo_df'])} repositories")
    return datasets


def annotate_pr_features(pr_df: pd.DataFrame, commit_details_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate PR dataframe with computed features."""
    df = pr_df.copy()

    # Compute turnaround time
    df['turnaround_time'] = (
        pd.to_datetime(df['closed_at']) - pd.to_datetime(df['created_at'])
    ).dt.total_seconds()

    # Outcome columns
    df['accepted'] = df['merged_at'].notna()
    df['rejected'] = (df['state'] == 'closed') & df['merged_at'].isna()

    # Code churn from commit details
    pr_size = commit_details_df.groupby('pr_id').agg({
        'additions': 'sum',
        'deletions': 'sum'
    }).reset_index()
    pr_size.columns = ['id', 'lines_added', 'lines_deleted']
    df = df.merge(pr_size, on='id', how='left')

    df['lines_added'] = df['lines_added'].fillna(0)
    df['lines_deleted'] = df['lines_deleted'].fillna(0)
    df['net_churn'] = df['lines_added'] - df['lines_deleted']
    df['total_churn'] = df['lines_added'] + df['lines_deleted']

    # Number of files changed
    pr_files = commit_details_df.groupby('pr_id')['filename'].nunique().reset_index()
    pr_files.columns = ['id', 'num_files_changed']
    df = df.merge(pr_files, on='id', how='left')
    df['num_files_changed'] = df['num_files_changed'].fillna(0)

    return df


def identify_closed_not_planned(pr_df: pd.DataFrame, timeline_df: pd.DataFrame) -> pd.Series:
    """
    Identify PRs that were likely closed as 'not planned'.

    Heuristics:
    1. PRs with specific labels: 'duplicate', 'stale', 'wontfix', 'invalid', 'spam'
    2. PRs with explicit markers in title: [duplicate], [stale], [spam]
    3. PRs with phrases like "closed as duplicate", "closing as stale" in timeline messages

    NOTE: The GitHub API field 'state_reason' (which contains 'not_planned' or 'completed')
    is NOT available in the AIDev dataset. This is a heuristic approximation.
    """
    pr_ids = pr_df['id'].unique()
    not_planned_pr_ids = set()

    # Labels that indicate "not planned" closures
    not_planned_labels = [
        'duplicate', 'stale', 'wontfix', "won't fix", 'invalid', 'spam',
        'abandoned', 'superseded', 'not planned', 'wont-fix', 'not-planned'
    ]

    # Check 'label' column for relevant labels (most reliable signal)
    if 'label' in timeline_df.columns:
        label_events = timeline_df[
            (timeline_df['pr_id'].isin(pr_ids)) &
            (timeline_df['event'] == 'labeled')
        ].copy()

        if len(label_events) > 0:
            def is_not_planned_label(label):
                if pd.isna(label):
                    return False
                label_lower = str(label).lower().strip()
                return any(kw in label_lower for kw in not_planned_labels)

            flagged_labels = label_events[label_events['label'].apply(is_not_planned_label)]
            if len(flagged_labels) > 0:
                not_planned_pr_ids.update(flagged_labels['pr_id'].unique())

    # Check close event messages for explicit closure reasons
    close_events = timeline_df[
        (timeline_df['pr_id'].isin(pr_ids)) &
        (timeline_df['event'] == 'closed')
    ].copy()

    # Phrases that indicate explicit "not planned" closure
    closure_phrases = [
        'closed as duplicate', 'closing as duplicate', 'marked as duplicate',
        'closed as stale', 'closing as stale', 'marked as stale',
        'closed as spam', 'closing as spam',
        'closed as not planned', 'closing as not planned',
        'superseded by', 'duplicate of #', 'duplicate of http'
    ]

    if len(close_events) > 0 and 'message' in close_events.columns:
        def has_closure_phrase(text):
            if pd.isna(text):
                return False
            text_lower = str(text).lower()
            return any(phrase in text_lower for phrase in closure_phrases)

        close_events['is_not_planned'] = close_events['message'].apply(has_closure_phrase)
        flagged_close = close_events[close_events['is_not_planned']]
        if len(flagged_close) > 0:
            not_planned_pr_ids.update(flagged_close['pr_id'].unique())

    # Check PR title for explicit markers (brackets indicate intentional marking)
    title_markers = ['[duplicate]', '[stale]', '[spam]', '[wontfix]', '[invalid]', '[superseded]']

    def check_title_markers(row):
        title = str(row.get('title', '')).lower()
        return any(marker in title for marker in title_markers)

    pr_title_flagged = pr_df.apply(check_title_markers, axis=1)

    # Combine all signals
    is_not_planned = (
        pr_df['id'].isin(not_planned_pr_ids) |
        pr_title_flagged
    )

    return is_not_planned


def filter_pr_data(
    pr_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    min_turnaround_seconds: int = 60,
    exclude_not_planned: bool = False,
    top_percent_lines: int = 0
) -> pd.DataFrame:
    """
    Apply filters to PR dataframe.

    Filters:
    1. Only closed PRs
    2. Exclude PRs closed in < min_turnaround_seconds
    3. Optionally exclude "closed as not planned" PRs
    4. Optionally exclude top N% PRs by lines modified (total_churn)

    Args:
        top_percent_lines: If > 0, exclude the top N% of PRs by total_churn.
                          Set to 0 to disable this filter (default).
    """
    df = pr_df.copy()

    initial_count = len(df)

    # Filter closed PRs only
    df = df[df['state'] == 'closed']
    print(f"  After filtering closed PRs: {len(df)} (removed {initial_count - len(df)})")

    # Filter out instant closures
    df = df[df['turnaround_time'] >= min_turnaround_seconds]
    print(f"  After filtering turnaround >= {min_turnaround_seconds}s: {len(df)}")

    # Optionally filter "not planned" PRs
    if exclude_not_planned:
        is_not_planned = identify_closed_not_planned(df, timeline_df)
        not_planned_count = is_not_planned.sum()
        df = df[~is_not_planned]
        print(f"  After excluding 'not planned' PRs: {len(df)} (removed {not_planned_count})")

    # Optionally filter top N% by lines modified (total_churn)
    if top_percent_lines > 0 and 'total_churn' in df.columns:
        before_count = len(df)
        df = filter_top_n_for_cols(df, ['total_churn'], filter_percent=top_percent_lines)
        print(f"  After excluding top {top_percent_lines}% by lines modified: {len(df)} (removed {before_count - len(df)})")

    return df


def get_repo_prs(pr_df: pd.DataFrame, repo_df: pd.DataFrame, repo_identifier: str) -> tuple:
    """Get PRs for a specific repository by name or ID."""
    # Try to match by full_name first
    repo_match = repo_df[repo_df['full_name'] == repo_identifier]

    if repo_match.empty:
        # Try partial match
        repo_match = repo_df[repo_df['full_name'].str.contains(repo_identifier, case=False, na=False)]

    if repo_match.empty:
        # Try by ID if numeric
        try:
            repo_id = int(repo_identifier)
            repo_match = repo_df[repo_df['id'] == repo_id]
        except ValueError:
            pass

    if repo_match.empty:
        raise ValueError(f"Repository '{repo_identifier}' not found in dataset")

    if len(repo_match) > 1:
        print(f"Multiple matches found for '{repo_identifier}':")
        for _, row in repo_match.iterrows():
            print(f"  - {row['full_name']} (id: {row['id']})")
        raise ValueError("Please provide a more specific repository name")

    repo_info = repo_match.iloc[0].to_dict()
    repo_prs = pr_df[pr_df['repo_id'] == repo_info['id']]

    return repo_prs, repo_info


def list_available_repos(pr_df: pd.DataFrame, repo_df: pd.DataFrame) -> pd.DataFrame:
    """List all repositories with PR counts."""
    pr_counts = pr_df.groupby('repo_id').agg({
        'id': 'count',
        'agent': lambda x: x.value_counts().to_dict()
    }).reset_index()
    pr_counts.columns = ['repo_id', 'pr_count', 'agent_distribution']

    repos = repo_df.merge(pr_counts, left_on='id', right_on='repo_id', how='inner')
    repos = repos.sort_values('pr_count', ascending=False)

    # Select available columns
    cols = ['full_name', 'id', 'pr_count']
    for col in ['language', 'license', 'stargazers_count', 'stars']:
        if col in repos.columns:
            cols.append(col)

    return repos[cols]


def compute_basic_stats(
    repo_pr_df: pd.DataFrame,
    commits_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    reviews_df: pd.DataFrame
) -> dict:
    """Compute basic repository statistics."""
    pr_ids = repo_pr_df['id'].unique()

    repo_commits = commits_df[commits_df['pr_id'].isin(pr_ids)]
    repo_comments = comments_df[comments_df['pr_id'].isin(pr_ids)]
    repo_reviews = reviews_df[reviews_df['pr_id'].isin(pr_ids)]

    total_prs = len(repo_pr_df)
    accepted_prs = repo_pr_df['accepted'].sum()
    rejected_prs = repo_pr_df['rejected'].sum()

    return {
        'total_prs': int(total_prs),
        'accepted_prs': int(accepted_prs),
        'rejected_prs': int(rejected_prs),
        'acceptance_rate': float(accepted_prs / total_prs * 100) if total_prs > 0 else 0,
        'rejection_rate': float(rejected_prs / total_prs * 100) if total_prs > 0 else 0,
        'num_commits': int(len(repo_commits)),
        'num_comments': int(len(repo_comments)),
        'num_reviews': int(len(repo_reviews)),
        'avg_commits_per_pr': float(repo_commits.groupby('pr_id').size().mean()) if len(repo_commits) > 0 else 0,
        'avg_comments_per_pr': float(repo_comments.groupby('pr_id').size().mean()) if len(repo_comments) > 0 else 0,
        'avg_reviews_per_pr': float(repo_reviews.groupby('pr_id').size().mean()) if len(repo_reviews) > 0 else 0,
    }


def compute_time_stats(repo_pr_df: pd.DataFrame) -> dict:
    """Compute turnaround time statistics for accepted and rejected PRs."""

    def stats_for_series(s: pd.Series, prefix: str) -> dict:
        if len(s) == 0:
            return {f'{prefix}_{stat}': None for stat in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75']}
        return {
            f'{prefix}_mean': float(s.mean()),
            f'{prefix}_median': float(s.median()),
            f'{prefix}_std': float(s.std()),
            f'{prefix}_min': float(s.min()),
            f'{prefix}_max': float(s.max()),
            f'{prefix}_q25': float(s.quantile(0.25)),
            f'{prefix}_q75': float(s.quantile(0.75)),
        }

    accepted = repo_pr_df[repo_pr_df['accepted'] == True]['turnaround_time']
    rejected = repo_pr_df[repo_pr_df['rejected'] == True]['turnaround_time']
    all_prs = repo_pr_df['turnaround_time']

    return {
        **stats_for_series(accepted, 'accepted_turnaround'),
        **stats_for_series(rejected, 'rejected_turnaround'),
        **stats_for_series(all_prs, 'all_turnaround'),
    }


def compute_churn_stats(repo_pr_df: pd.DataFrame) -> dict:
    """Compute code churn statistics."""

    def stats_for_col(df: pd.DataFrame, col: str) -> dict:
        s = df[col].dropna()
        if len(s) == 0:
            return {f'{col}_{stat}': None for stat in ['mean', 'median', 'std', 'min', 'max']}
        return {
            f'{col}_mean': float(s.mean()),
            f'{col}_median': float(s.median()),
            f'{col}_std': float(s.std()),
            f'{col}_min': float(s.min()),
            f'{col}_max': float(s.max()),
        }

    result = {}
    for col in ['lines_added', 'lines_deleted', 'net_churn', 'total_churn', 'num_files_changed']:
        if col in repo_pr_df.columns:
            result.update(stats_for_col(repo_pr_df, col))

    return result


def compute_agent_stats(repo_pr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute acceptance/rejection rate per AI agent."""
    agent_stats = repo_pr_df.groupby('agent').agg({
        'id': 'count',
        'accepted': ['sum', 'mean'],
        'rejected': ['sum', 'mean'],
        'turnaround_time': ['mean', 'median'],
    }).reset_index()

    agent_stats.columns = [
        'agent', 'total_prs',
        'accepted_count', 'acceptance_rate',
        'rejected_count', 'rejection_rate',
        'mean_turnaround', 'median_turnaround'
    ]

    agent_stats['acceptance_rate'] = agent_stats['acceptance_rate'] * 100
    agent_stats['rejection_rate'] = agent_stats['rejection_rate'] * 100

    return agent_stats.sort_values('total_prs', ascending=False)


def compute_task_type_stats(repo_pr_df: pd.DataFrame, task_type_df: pd.DataFrame) -> pd.DataFrame:
    """Compute acceptance rate per task type."""
    # Column is 'type' in the dataset, not 'task_type'
    pr_with_task = repo_pr_df.merge(
        task_type_df[['id', 'type']],
        on='id',
        how='left'
    )
    pr_with_task = pr_with_task.rename(columns={'type': 'task_type'})

    task_stats = pr_with_task.groupby('task_type').agg({
        'id': 'count',
        'accepted': ['sum', 'mean'],
        'rejected': ['sum', 'mean'],
    }).reset_index()

    task_stats.columns = [
        'task_type', 'total_prs',
        'accepted_count', 'acceptance_rate',
        'rejected_count', 'rejection_rate'
    ]

    task_stats['acceptance_rate'] = task_stats['acceptance_rate'] * 100
    task_stats['rejection_rate'] = task_stats['rejection_rate'] * 100

    # Sort by task type order
    task_stats['sort_key'] = task_stats['task_type'].apply(
        lambda x: TASK_TYPE_ORDER.index(x) if x in TASK_TYPE_ORDER else len(TASK_TYPE_ORDER)
    )
    task_stats = task_stats.sort_values('sort_key').drop('sort_key', axis=1)

    return task_stats


def compute_user_activity(
    repo_pr_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    reviews_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute user activity statistics."""
    pr_ids = repo_pr_df['id'].unique()

    # PR authors
    pr_authors = repo_pr_df.groupby('user').agg({
        'id': 'count',
        'accepted': 'sum',
    }).reset_index()
    pr_authors.columns = ['user', 'prs_authored', 'prs_accepted']

    # Comments
    repo_comments = comments_df[comments_df['pr_id'].isin(pr_ids)]
    comment_counts = repo_comments.groupby('user').size().reset_index(name='comments_made')

    # Reviews
    repo_reviews = reviews_df[reviews_df['pr_id'].isin(pr_ids)]
    review_counts = repo_reviews.groupby('user').size().reset_index(name='reviews_made')

    # Merge all
    user_activity = pr_authors.merge(comment_counts, on='user', how='outer')
    user_activity = user_activity.merge(review_counts, on='user', how='outer')
    user_activity = user_activity.fillna(0)

    # Total activity score
    user_activity['total_activity'] = (
        user_activity['prs_authored'] +
        user_activity['comments_made'] +
        user_activity['reviews_made']
    )

    return user_activity.sort_values('total_activity', ascending=False)


def identify_outlier_prs(repo_pr_df: pd.DataFrame) -> pd.DataFrame:
    """Identify PRs with extreme values using IQR method."""
    columns = ['turnaround_time', 'total_churn', 'num_files_changed']
    columns = [c for c in columns if c in repo_pr_df.columns]

    outliers = []

    for col in columns:
        data = repo_pr_df[col].dropna()
        if len(data) < 4:
            continue

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        col_outliers = repo_pr_df[
            (repo_pr_df[col] < lower) | (repo_pr_df[col] > upper)
        ].copy()
        col_outliers['outlier_column'] = col
        col_outliers['outlier_type'] = col_outliers[col].apply(
            lambda x: 'below' if x < lower else 'above'
        )
        col_outliers['threshold'] = col_outliers['outlier_type'].apply(
            lambda x: lower if x == 'below' else upper
        )
        outliers.append(col_outliers)

    if not outliers:
        return pd.DataFrame()

    all_outliers = pd.concat(outliers, ignore_index=True)
    return all_outliers[['id', 'number', 'title', 'outlier_column', 'outlier_type',
                         'turnaround_time', 'total_churn', 'num_files_changed', 'accepted']].drop_duplicates()


def mann_whitney_cliff_delta(dist1: pd.Series, dist2: pd.Series) -> dict:
    """Perform Mann-Whitney U test and compute Cliff's delta."""
    d1 = dist1.dropna()
    d2 = dist2.dropna()

    if len(d1) < 2 or len(d2) < 2:
        return {'u_statistic': None, 'p_value': None, 'cliffs_delta': None, 'effect_size': None}

    u, p = stats.mannwhitneyu(d1, d2, alternative="two-sided")

    if cliffs_delta:
        d, size = cliffs_delta(d1.tolist(), d2.tolist())
    else:
        # Manual Cliff's delta calculation
        n1, n2 = len(d1), len(d2)
        more = sum(1 for x in d1 for y in d2 if x > y)
        less = sum(1 for x in d1 for y in d2 if x < y)
        d = (more - less) / (n1 * n2)

        abs_d = abs(d)
        if abs_d < 0.147:
            size = 'negligible'
        elif abs_d < 0.33:
            size = 'small'
        elif abs_d < 0.474:
            size = 'medium'
        else:
            size = 'large'

    return {
        'u_statistic': float(u),
        'p_value': float(p),
        'cliffs_delta': float(d),
        'effect_size': size
    }


# ============ Visualization Functions ============

def create_turnaround_distribution(repo_pr_df: pd.DataFrame, output_dir: Path) -> None:
    """Create turnaround time distribution visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    accepted = repo_pr_df[repo_pr_df['accepted'] == True]['turnaround_time'] / 3600  # Convert to hours
    rejected = repo_pr_df[repo_pr_df['rejected'] == True]['turnaround_time'] / 3600

    # Histogram
    ax = axes[0]
    bins = np.linspace(0, min(accepted.quantile(0.95), rejected.quantile(0.95) if len(rejected) > 0 else accepted.quantile(0.95)), 50)

    if len(accepted) > 0:
        ax.hist(accepted, bins=bins, alpha=0.7, label=f'Accepted (n={len(accepted)})',
                color=OUTCOME_COLORS['accepted'], edgecolor='black', linewidth=0.5)
    if len(rejected) > 0:
        ax.hist(rejected, bins=bins, alpha=0.7, label=f'Rejected (n={len(rejected)})',
                color=OUTCOME_COLORS['rejected'], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Turnaround Time (hours)')
    ax.set_ylabel('Count')
    ax.set_title('Turnaround Time Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Box plot
    ax = axes[1]
    data_to_plot = []
    labels = []
    colors = []

    if len(accepted) > 0:
        data_to_plot.append(accepted)
        labels.append('Accepted')
        colors.append(OUTCOME_COLORS['accepted'])
    if len(rejected) > 0:
        data_to_plot.append(rejected)
        labels.append('Rejected')
        colors.append(OUTCOME_COLORS['rejected'])

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_ylabel('Turnaround Time (hours)')
    ax.set_title('Turnaround Time Comparison')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'turnaround_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'turnaround_distribution.pdf', bbox_inches='tight')
    plt.close()


def create_churn_distribution(repo_pr_df: pd.DataFrame, output_dir: Path) -> None:
    """Create code changes distribution visualization (without outliers for better readability)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Renamed metrics with clearer, more descriptive names
    metrics = [
        ('lines_added', 'Lines Added'),
        ('lines_deleted', 'Lines Removed'),
        ('total_churn', 'Total Code Changes\n(Added + Removed)'),
        ('num_files_changed', 'Files Modified')
    ]

    for ax, (col, title) in zip(axes.flatten(), metrics):
        if col not in repo_pr_df.columns:
            ax.text(0.5, 0.5, f'{title}\nNot available', ha='center', va='center')
            continue

        accepted = repo_pr_df[repo_pr_df['accepted'] == True][col].dropna()
        rejected = repo_pr_df[repo_pr_df['rejected'] == True][col].dropna()

        data_to_plot = []
        labels = []
        colors = []

        if len(accepted) > 0:
            data_to_plot.append(accepted)
            labels.append('Accepted')
            colors.append(OUTCOME_COLORS['accepted'])
        if len(rejected) > 0:
            data_to_plot.append(rejected)
            labels.append('Rejected')
            colors.append(OUTCOME_COLORS['rejected'])

        if data_to_plot:
            # Hide outliers with showfliers=False for cleaner visualization
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=False)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Set Y axis limit to 95th percentile to focus on the main distribution
            all_data = pd.concat([d for d in data_to_plot])
            upper_limit = all_data.quantile(0.95)
            # Add 10% padding above the 95th percentile
            ax.set_ylim(0, upper_limit * 1.1)

        ax.set_title(title)
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)

    # Add a note about outliers being excluded
    fig.text(0.5, 0.01, 'Note: Outliers excluded for better visualization (showing up to 95th percentile)', 
             ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_dir / 'code_changes_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'code_changes_distribution.pdf', bbox_inches='tight')
    # Also save with old name for backward compatibility
    plt.savefig(output_dir / 'churn_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'churn_distribution.pdf', bbox_inches='tight')
    plt.close()


def create_agent_breakdown(agent_stats: pd.DataFrame, output_dir: Path) -> None:
    """Create agent breakdown visualization."""
    if agent_stats.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar chart
    ax = axes[0]
    agents = agent_stats['agent'].tolist()
    x = np.arange(len(agents))
    width = 0.6

    colors = [COLOR_MAP.get(a, '#808080') for a in agents]

    accepted = agent_stats['accepted_count'].values
    rejected = agent_stats['rejected_count'].values

    ax.bar(x, accepted, width, label='Accepted', color=OUTCOME_COLORS['accepted'], edgecolor='black', linewidth=0.5)
    ax.bar(x, rejected, width, bottom=accepted, label='Rejected', color=OUTCOME_COLORS['rejected'], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Agent')
    ax.set_ylabel('Number of PRs')
    ax.set_title('PRs by Agent and Outcome')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Acceptance rate
    ax = axes[1]
    bars = ax.bar(x, agent_stats['acceptance_rate'].values, width, color=colors, edgecolor='black', linewidth=0.5)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Agent')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Acceptance Rate by Agent')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, agent_stats['acceptance_rate'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'agent_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'agent_breakdown.pdf', bbox_inches='tight')
    plt.close()


def create_task_type_breakdown(task_type_stats: pd.DataFrame, output_dir: Path) -> None:
    """Create task type breakdown visualization."""
    if task_type_stats.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count by task type
    ax = axes[0]
    task_types = task_type_stats['task_type'].tolist()
    x = np.arange(len(task_types))
    width = 0.6

    accepted = task_type_stats['accepted_count'].values
    rejected = task_type_stats['rejected_count'].values

    ax.bar(x, accepted, width, label='Accepted', color=OUTCOME_COLORS['accepted'], edgecolor='black', linewidth=0.5)
    ax.bar(x, rejected, width, bottom=accepted, label='Rejected', color=OUTCOME_COLORS['rejected'], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Task Type')
    ax.set_ylabel('Number of PRs')
    ax.set_title('PRs by Task Type and Outcome')
    ax.set_xticks(x)
    ax.set_xticklabels(task_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Acceptance rate by task type
    ax = axes[1]
    bars = ax.bar(x, task_type_stats['acceptance_rate'].values, width,
                  color='#3498db', edgecolor='black', linewidth=0.5)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Task Type')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Acceptance Rate by Task Type')
    ax.set_xticks(x)
    ax.set_xticklabels(task_types, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'task_type_acceptance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'task_type_acceptance.pdf', bbox_inches='tight')
    plt.close()


def create_pr_timeline(repo_pr_df: pd.DataFrame, output_dir: Path) -> None:
    """Create PR timeline visualization."""
    fig, ax = plt.subplots(figsize=(14, 5))

    df = repo_pr_df.copy()
    df['created_date'] = pd.to_datetime(df['created_at']).dt.date

    daily_counts = df.groupby(['created_date', 'accepted']).size().unstack(fill_value=0)

    if True in daily_counts.columns:
        ax.fill_between(daily_counts.index, 0, daily_counts[True],
                       alpha=0.7, color=OUTCOME_COLORS['accepted'], label='Accepted')
    if False in daily_counts.columns:
        bottom = daily_counts[True] if True in daily_counts.columns else 0
        ax.fill_between(daily_counts.index, bottom, bottom + daily_counts[False],
                       alpha=0.7, color=OUTCOME_COLORS['rejected'], label='Rejected')

    ax.set_xlabel('Date')
    ax.set_ylabel('Number of PRs')
    ax.set_title('PR Activity Over Time')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'pr_timeline.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'pr_timeline.pdf', bbox_inches='tight')
    plt.close()


def create_user_activity_chart(user_activity: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    """Create user activity visualization."""
    if user_activity.empty:
        return

    top_users = user_activity.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    y = np.arange(len(top_users))
    height = 0.25

    ax.barh(y - height, top_users['prs_authored'], height, label='PRs Authored', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.barh(y, top_users['comments_made'], height, label='Comments', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.barh(y + height, top_users['reviews_made'], height, label='Reviews', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Count')
    ax.set_ylabel('User')
    ax.set_title(f'Top {top_n} Most Active Users')
    ax.set_yticks(y)
    ax.set_yticklabels(top_users['user'].tolist())
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'user_activity.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'user_activity.pdf', bbox_inches='tight')
    plt.close()


def generate_report(
    repo_info: dict,
    basic_stats: dict,
    time_stats: dict,
    churn_stats: dict,
    agent_stats: pd.DataFrame,
    task_type_stats: pd.DataFrame,
    user_activity: pd.DataFrame,
    outlier_prs: pd.DataFrame,
    statistical_tests: dict = None,
    output_dir: Path = None
) -> str:
    """Generate markdown report."""

    def format_time(seconds):
        if seconds is None:
            return "N/A"
        hours = seconds / 3600
        if hours < 1:
            return f"{seconds/60:.1f} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        else:
            return f"{hours/24:.1f} days"

    report = []
    report.append(f"# Repository Analysis Report: {repo_info.get('full_name', 'Unknown')}")
    report.append("")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Repository Info
    report.append("## Repository Information")
    report.append("")
    report.append(f"- **Name:** {repo_info.get('full_name', 'N/A')}")
    report.append(f"- **Language:** {repo_info.get('language', 'N/A')}")
    report.append(f"- **License:** {repo_info.get('license', 'N/A')}")
    report.append(f"- **Stars:** {repo_info.get('stargazers_count', 'N/A')}")
    report.append(f"- **Forks:** {repo_info.get('forks_count', 'N/A')}")
    report.append("")

    # Basic Statistics
    report.append("## Pull Request Statistics")
    report.append("")
    report.append(f"- **Total PRs:** {basic_stats['total_prs']}")
    report.append(f"- **Accepted PRs:** {basic_stats['accepted_prs']} ({basic_stats['acceptance_rate']:.1f}%)")
    report.append(f"- **Rejected PRs:** {basic_stats['rejected_prs']} ({basic_stats['rejection_rate']:.1f}%)")
    report.append(f"- **Total Commits:** {basic_stats['num_commits']}")
    report.append(f"- **Total Comments:** {basic_stats['num_comments']}")
    report.append(f"- **Total Reviews:** {basic_stats['num_reviews']}")
    report.append(f"- **Avg Commits/PR:** {basic_stats['avg_commits_per_pr']:.1f}")
    report.append(f"- **Avg Comments/PR:** {basic_stats['avg_comments_per_pr']:.1f}")
    report.append(f"- **Avg Reviews/PR:** {basic_stats['avg_reviews_per_pr']:.1f}")
    report.append("")

    # Time Statistics
    report.append("## Turnaround Time Statistics")
    report.append("")
    report.append("### Accepted PRs")
    report.append(f"- Mean: {format_time(time_stats.get('accepted_turnaround_mean'))}")
    report.append(f"- Median: {format_time(time_stats.get('accepted_turnaround_median'))}")
    report.append(f"- Min: {format_time(time_stats.get('accepted_turnaround_min'))}")
    report.append(f"- Max: {format_time(time_stats.get('accepted_turnaround_max'))}")
    report.append("")
    report.append("### Rejected PRs")
    report.append(f"- Mean: {format_time(time_stats.get('rejected_turnaround_mean'))}")
    report.append(f"- Median: {format_time(time_stats.get('rejected_turnaround_median'))}")
    report.append(f"- Min: {format_time(time_stats.get('rejected_turnaround_min'))}")
    report.append(f"- Max: {format_time(time_stats.get('rejected_turnaround_max'))}")
    report.append("")

    # Code Churn
    report.append("## Code Churn Statistics")
    report.append("")
    report.append(f"- **Mean Lines Added:** {churn_stats.get('lines_added_mean', 0):.1f}")
    report.append(f"- **Median Lines Added:** {churn_stats.get('lines_added_median', 0):.1f}")
    report.append(f"- **Mean Lines Deleted:** {churn_stats.get('lines_deleted_mean', 0):.1f}")
    report.append(f"- **Median Lines Deleted:** {churn_stats.get('lines_deleted_median', 0):.1f}")
    report.append(f"- **Mean Total Churn:** {churn_stats.get('total_churn_mean', 0):.1f}")
    report.append(f"- **Mean Files Changed:** {churn_stats.get('num_files_changed_mean', 0):.1f}")
    report.append("")

    # Agent Breakdown
    if not agent_stats.empty:
        report.append("## Agent Breakdown")
        report.append("")
        report.append("| Agent | Total PRs | Accepted | Rejected | Acceptance Rate |")
        report.append("|-------|-----------|----------|----------|-----------------|")
        for _, row in agent_stats.iterrows():
            report.append(f"| {row['agent']} | {int(row['total_prs'])} | {int(row['accepted_count'])} | {int(row['rejected_count'])} | {row['acceptance_rate']:.1f}% |")
        report.append("")

    # Task Type Breakdown
    if not task_type_stats.empty:
        report.append("## Task Type Breakdown")
        report.append("")
        report.append("| Task Type | Total PRs | Accepted | Rejected | Acceptance Rate |")
        report.append("|-----------|-----------|----------|----------|-----------------|")
        for _, row in task_type_stats.iterrows():
            report.append(f"| {row['task_type']} | {int(row['total_prs'])} | {int(row['accepted_count'])} | {int(row['rejected_count'])} | {row['acceptance_rate']:.1f}% |")
        report.append("")

    # Top Users
    if not user_activity.empty:
        report.append("## Top 10 Active Users")
        report.append("")
        report.append("| User | PRs Authored | Comments | Reviews | Total Activity |")
        report.append("|------|--------------|----------|---------|----------------|")
        for _, row in user_activity.head(10).iterrows():
            report.append(f"| {row['user']} | {int(row['prs_authored'])} | {int(row['comments_made'])} | {int(row['reviews_made'])} | {int(row['total_activity'])} |")
        report.append("")

    # Outliers
    if not outlier_prs.empty:
        report.append("## Outlier PRs")
        report.append("")
        report.append(f"Found {len(outlier_prs)} PRs with extreme values:")
        report.append("")
        for _, row in outlier_prs.head(10).iterrows():
            outcome = "Accepted" if row['accepted'] else "Rejected"
            report.append(f"- PR #{int(row['number'])}: {row['title'][:50]}... ({row['outlier_column']} {row['outlier_type']}) - {outcome}")
        report.append("")

    # Statistical Tests
    if statistical_tests:
        report.append("## Statistical Tests (Accepted vs Rejected)")
        report.append("")
        for metric, results in statistical_tests.items():
            if results['p_value'] is not None:
                sig = "**" if results['p_value'] < 0.05 else ""
                report.append(f"### {metric}")
                report.append(f"- Mann-Whitney U: {results['u_statistic']:.2f}")
                report.append(f"- p-value: {sig}{results['p_value']:.4f}{sig}")
                report.append(f"- Cliff's delta: {results['cliffs_delta']:.3f} ({results['effect_size']})")
                report.append("")

    return "\n".join(report)


def analyze_repository(
    repo_identifier: str,
    output_base_dir: str = "./specific_repo_exploration",
    min_turnaround: int = 60,
    include_not_planned: bool = False,
    statistical_tests: bool = False,
    exclude_top_churn: int = 0
) -> None:
    """Main function to analyze a repository."""

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
        exclude_not_planned=not include_not_planned,
        top_percent_lines=exclude_top_churn
    )
    print(f"After filtering: {len(repo_pr_df)} PRs")

    if len(repo_pr_df) == 0:
        print("No PRs remaining after filtering. Exiting.")
        return

    # Create output directory
    repo_name_safe = repo_info['full_name'].replace('/', '_')
    output_dir = Path(output_base_dir) / repo_name_safe
    viz_dir = output_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    print("Computing statistics...")
    basic_stats = compute_basic_stats(
        repo_pr_df,
        datasets['pr_commits_df'],
        datasets['pr_comments_df'],
        datasets['pr_reviews_df']
    )
    time_stats = compute_time_stats(repo_pr_df)
    churn_stats = compute_churn_stats(repo_pr_df)
    agent_stats = compute_agent_stats(repo_pr_df)
    task_type_stats = compute_task_type_stats(repo_pr_df, datasets['pr_task_type_df'])
    user_activity = compute_user_activity(
        repo_pr_df,
        datasets['pr_comments_df'],
        datasets['pr_reviews_df']
    )
    outlier_prs = identify_outlier_prs(repo_pr_df)

    # Statistical tests
    stat_test_results = None
    if statistical_tests:
        print("Running statistical tests...")
        stat_test_results = {}
        accepted = repo_pr_df[repo_pr_df['accepted'] == True]
        rejected = repo_pr_df[repo_pr_df['rejected'] == True]

        for col in ['turnaround_time', 'total_churn', 'lines_added', 'num_files_changed']:
            if col in repo_pr_df.columns:
                stat_test_results[col] = mann_whitney_cliff_delta(accepted[col], rejected[col])

    # Save statistics
    print("Saving statistics...")

    # JSON summary
    all_stats = {
        'repository': repo_info,
        'basic_stats': basic_stats,
        'time_stats': time_stats,
        'churn_stats': churn_stats,
    }
    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)

    # CSV files
    pd.DataFrame([basic_stats]).to_csv(output_dir / 'basic_stats.csv', index=False)
    pd.DataFrame([time_stats]).to_csv(output_dir / 'time_stats.csv', index=False)
    pd.DataFrame([churn_stats]).to_csv(output_dir / 'churn_stats.csv', index=False)
    agent_stats.to_csv(output_dir / 'agent_breakdown.csv', index=False)
    task_type_stats.to_csv(output_dir / 'task_type_stats.csv', index=False)
    user_activity.to_csv(output_dir / 'user_activity.csv', index=False)
    if not outlier_prs.empty:
        outlier_prs.to_csv(output_dir / 'outlier_prs.csv', index=False)

    # Create visualizations
    print("Creating visualizations...")
    create_turnaround_distribution(repo_pr_df, viz_dir)
    create_churn_distribution(repo_pr_df, viz_dir)
    create_agent_breakdown(agent_stats, viz_dir)
    create_task_type_breakdown(task_type_stats, viz_dir)
    create_pr_timeline(repo_pr_df, viz_dir)
    create_user_activity_chart(user_activity, viz_dir)

    # Generate report
    print("Generating report...")
    report = generate_report(
        repo_info, basic_stats, time_stats, churn_stats,
        agent_stats, task_type_stats, user_activity, outlier_prs,
        stat_test_results, output_dir
    )
    with open(output_dir / 'report.md', 'w') as f:
        f.write(report)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"  - summary_stats.json")
    print(f"  - report.md")
    print(f"  - visualizations/")


def main():
    parser = argparse.ArgumentParser(
        description="Repository-level PR analysis for AIDev dataset"
    )

    # Repository selection
    repo_group = parser.add_mutually_exclusive_group()
    repo_group.add_argument("--repo", "-r", type=str,
                           help="Repository full name (e.g., 'owner/repo')")
    repo_group.add_argument("--repo-id", type=int,
                           help="Repository ID")
    repo_group.add_argument("--list-repos", action="store_true",
                           help="List available repositories")

    # Output options
    parser.add_argument("--output", "-o", type=str,
                       default="./specific_repo_exploration",
                       help="Output base directory")

    # Filtering options
    parser.add_argument("--min-turnaround", type=int, default=60,
                       help="Minimum turnaround time in seconds (default: 60)")
    parser.add_argument("--include-not-planned", action="store_true",
                       help="Include PRs that were 'closed as not planned'")
    parser.add_argument("--exclude-top-churn", type=int, default=10,
                       help="Exclude top N%% of PRs by lines modified (default: 10)")

    # Analysis options
    parser.add_argument("--statistical-tests", action="store_true",
                       help="Include Mann-Whitney U and Cliff's delta tests")

    args = parser.parse_args()

    if args.list_repos:
        datasets = load_datasets()
        pr_df = annotate_pr_features(datasets['pr_df'], datasets['pr_commit_details_df'])
        repos = list_available_repos(pr_df, datasets['repo_df'])
        print("\nAvailable repositories:")
        print(repos.to_string(index=False))
        return

    if not args.repo and not args.repo_id:
        parser.print_help()
        print("\nError: Please specify --repo, --repo-id, or --list-repos")
        return

    repo_identifier = args.repo if args.repo else str(args.repo_id)

    analyze_repository(
        repo_identifier=repo_identifier,
        output_base_dir=args.output,
        min_turnaround=args.min_turnaround,
        include_not_planned=args.include_not_planned,
        statistical_tests=args.statistical_tests,
        exclude_top_churn=args.exclude_top_churn
    )


if __name__ == "__main__":
    main()
