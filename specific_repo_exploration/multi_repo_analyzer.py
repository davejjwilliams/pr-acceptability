#!/usr/bin/env python3
"""
Multi-Repository Analyzer for AIDev Dataset

Analyzes pull request statistics for top N repositories by stars in the AIDev dataset.
Generates comparative visualizations and reports across multiple repositories.

Usage:
    uv run specific_repo_exploration/multi_repo_analyzer.py
    uv run specific_repo_exploration/multi_repo_analyzer.py --num-repos 15
    uv run specific_repo_exploration/multi_repo_analyzer.py -n 10 -o ./output
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import filter_top_n_for_cols

# ============ Constants ============

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
    "open": "#95A5A6",
}

TASK_TYPE_ORDER = [
    "feat", "fix", "perf", "refactor", "style",
    "docs", "test", "chore", "build", "ci", "revert", "other"
]

AGENTS = ["Human", "OpenAI_Codex", "Devin", "Copilot", "Cursor", "Claude_Code"]

BOT_PATTERNS = [
    r'\[bot\]',
    r'(?i)_ai$',
    r'(?i)-ai$',
    r'(?i)^ai_',
    r'(?i)^ai-',
    r'(?i)bot$',
    r'(?i)_bot$',
    r'(?i)-bot$',
    r'(?i)^bot_',
    r'(?i)^bot-',
    r'(?i)dependabot',
    r'(?i)renovate',
    r'(?i)github-actions',
    r'(?i)codecov',
    r'(?i)semantic-release',
    r'(?i)greenkeeper',
]


# ============ Data Loading ============

def load_datasets() -> dict:
    """Load all required datasets from HuggingFace."""
    print("Loading datasets from HuggingFace...")
    datasets = {
        'pr_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet"),
        'repo_df': pd.read_parquet("hf://datasets/hao-li/AIDev/repository.parquet"),
        'pr_comments_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_comments.parquet"),
        'pr_reviews_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_reviews.parquet"),
        'pr_commit_details_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_commit_details.parquet"),
        'pr_task_type_df': pd.read_parquet("hf://datasets/hao-li/AIDev/pr_task_type.parquet"),
        'user_df': pd.read_parquet("hf://datasets/hao-li/AIDev/user.parquet"),
    }
    print(f"Loaded {len(datasets['pr_df'])} PRs from {len(datasets['repo_df'])} repositories")
    return datasets


def get_top_n_repos_by_stars(repo_df: pd.DataFrame, pr_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Select top N repositories by stars that have closed PRs in the dataset."""
    # Get repos that have CLOSED PRs (not just any PRs)
    closed_prs = pr_df[pr_df['state'] == 'closed']
    repos_with_closed_prs = closed_prs['repo_id'].unique()

    # Filter repo_df to only repos with closed PRs
    filtered_repos = repo_df[repo_df['id'].isin(repos_with_closed_prs)].copy()

    # Sort by stars and get top N
    star_col = 'stargazers_count' if 'stargazers_count' in filtered_repos.columns else 'stars'
    top_repos = filtered_repos.nlargest(n, star_col)

    print(f"Selected top {len(top_repos)} repositories by stars (with closed PRs):")
    for _, row in top_repos.iterrows():
        print(f"  - {row['full_name']}: {row[star_col]:,} stars")

    return top_repos


# ============ Feature Annotation ============

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
    df['is_open'] = df['state'] == 'open'

    # Code churn from commit details
    pr_size = commit_details_df.groupby('pr_id').agg({
        'additions': 'sum',
        'deletions': 'sum'
    }).reset_index()
    pr_size.columns = ['id', 'lines_added', 'lines_deleted']
    df = df.merge(pr_size, on='id', how='left')

    df['lines_added'] = df['lines_added'].fillna(0)
    df['lines_deleted'] = df['lines_deleted'].fillna(0)
    df['total_churn'] = df['lines_added'] + df['lines_deleted']

    # Number of files changed
    pr_files = commit_details_df.groupby('pr_id')['filename'].nunique().reset_index()
    pr_files.columns = ['id', 'num_files_changed']
    df = df.merge(pr_files, on='id', how='left')
    df['num_files_changed'] = df['num_files_changed'].fillna(0)

    return df


def apply_filters(
    pr_df: pd.DataFrame,
    min_turnaround: int = 60,
    top_percent_lines: int = 10
) -> pd.DataFrame:
    """Apply filters: min turnaround time and top N% lines modified exclusion."""
    df = pr_df.copy()
    initial_count = len(df)

    # Filter 1: Exclude PRs closed in less than min_turnaround seconds
    # Only apply to closed PRs
    closed_mask = df['state'] == 'closed'
    quick_close_mask = closed_mask & (df['turnaround_time'] < min_turnaround)
    df = df[~quick_close_mask]
    after_turnaround = len(df)
    print(f"Filtered {initial_count - after_turnaround} PRs closed in < {min_turnaround}s")

    # Filter 2: Exclude top N% by total_churn (lines modified)
    if top_percent_lines > 0 and 'total_churn' in df.columns:
        df = filter_top_n_for_cols(df, ['total_churn'], filter_percent=top_percent_lines)
        print(f"After filtering top {top_percent_lines}%: {len(df)} PRs remaining")

    return df


# ============ Statistics Functions ============

def compute_pr_status_counts(pr_df: pd.DataFrame, repo_df: pd.DataFrame) -> pd.DataFrame:
    """Compute PR status counts per repository."""
    results = []

    for _, repo in repo_df.iterrows():
        repo_prs = pr_df[pr_df['repo_id'] == repo['id']]

        accepted = repo_prs['accepted'].sum()
        rejected = repo_prs['rejected'].sum()
        is_open = repo_prs['is_open'].sum() if 'is_open' in repo_prs.columns else 0

        results.append({
            'repo_name': repo['full_name'],
            'repo_id': repo['id'],
            'accepted': int(accepted),
            'rejected': int(rejected),
            'open': int(is_open),
            'total': len(repo_prs)
        })

    return pd.DataFrame(results)


def compute_agent_counts(pr_df: pd.DataFrame, repo_df: pd.DataFrame) -> pd.DataFrame:
    """Compute PR counts by agent per repository."""
    results = []

    for _, repo in repo_df.iterrows():
        repo_prs = pr_df[pr_df['repo_id'] == repo['id']]

        row = {'repo_name': repo['full_name'], 'repo_id': repo['id']}
        total = 0

        for agent in AGENTS:
            if 'agent' in repo_prs.columns:
                count = len(repo_prs[repo_prs['agent'] == agent])
            else:
                count = 0
            row[agent] = count
            total += count

        row['Total'] = total if total > 0 else len(repo_prs)
        results.append(row)

    return pd.DataFrame(results)


def compute_turnaround_data(pr_df: pd.DataFrame, repo_df: pd.DataFrame) -> dict:
    """Compute turnaround time data per repository."""
    data = {}

    for _, repo in repo_df.iterrows():
        repo_prs = pr_df[pr_df['repo_id'] == repo['id']]

        accepted_time = repo_prs[repo_prs['accepted'] == True]['turnaround_time'] / 3600  # hours
        rejected_time = repo_prs[repo_prs['rejected'] == True]['turnaround_time'] / 3600

        data[repo['full_name']] = {
            'accepted': accepted_time.dropna(),
            'rejected': rejected_time.dropna()
        }

    return data


def compute_churn_data(pr_df: pd.DataFrame, repo_df: pd.DataFrame) -> dict:
    """Compute code churn data per repository."""
    data = {}

    for _, repo in repo_df.iterrows():
        repo_prs = pr_df[pr_df['repo_id'] == repo['id']]

        data[repo['full_name']] = {
            'lines_added': repo_prs['lines_added'].dropna(),
            'lines_deleted': repo_prs['lines_deleted'].dropna(),
            'num_files_changed': repo_prs['num_files_changed'].dropna()
        }

    return data


def compute_task_type_acceptance(
    pr_df: pd.DataFrame,
    task_type_df: pd.DataFrame,
    repo_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute acceptance rate by task type per repository."""
    results = []

    # The task_type_df uses 'id' for PR id and 'type' for task type
    pr_with_task = pr_df.merge(
        task_type_df[['id', 'type']].rename(columns={'id': 'task_pr_id', 'type': 'task_type'}),
        left_on='id',
        right_on='task_pr_id',
        how='left'
    )
    pr_with_task['task_type'] = pr_with_task['task_type'].fillna('other')

    for _, repo in repo_df.iterrows():
        repo_prs = pr_with_task[pr_with_task['repo_id'] == repo['id']]

        for task_type in TASK_TYPE_ORDER:
            task_prs = repo_prs[repo_prs['task_type'] == task_type]
            if len(task_prs) > 0:
                accepted = task_prs['accepted'].sum()
                rejected = task_prs['rejected'].sum()
                total = len(task_prs)
                acceptance_rate = (accepted / total * 100) if total > 0 else 0

                results.append({
                    'repo_name': repo['full_name'],
                    'task_type': task_type,
                    'accepted': int(accepted),
                    'rejected': int(rejected),
                    'total': total,
                    'acceptance_rate': acceptance_rate
                })

    return pd.DataFrame(results)


def compute_review_comment_data(
    pr_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    repo_df: pd.DataFrame
) -> dict:
    """Compute review and comment counts per PR by outcome per repository."""
    # Count reviews per PR
    review_counts = reviews_df.groupby('pr_id').size().reset_index(name='num_reviews')

    # Count comments per PR
    comment_counts = comments_df.groupby('pr_id').size().reset_index(name='num_comments')

    # Merge with PR data
    pr_with_counts = pr_df.merge(review_counts, left_on='id', right_on='pr_id', how='left')
    pr_with_counts = pr_with_counts.merge(comment_counts, left_on='id', right_on='pr_id', how='left')
    pr_with_counts['num_reviews'] = pr_with_counts['num_reviews'].fillna(0)
    pr_with_counts['num_comments'] = pr_with_counts['num_comments'].fillna(0)

    data = {}

    for _, repo in repo_df.iterrows():
        repo_prs = pr_with_counts[pr_with_counts['repo_id'] == repo['id']]

        accepted_prs = repo_prs[repo_prs['accepted'] == True]
        rejected_prs = repo_prs[repo_prs['rejected'] == True]

        data[repo['full_name']] = {
            'accepted_reviews': accepted_prs['num_reviews'],
            'rejected_reviews': rejected_prs['num_reviews'],
            'accepted_comments': accepted_prs['num_comments'],
            'rejected_comments': rejected_prs['num_comments'],
        }

    return data


def identify_bot_users(
    pr_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    repo_df: pd.DataFrame,
    top_percent: float = 15
) -> pd.DataFrame:
    """Identify bot/AI users among top active users in selected repositories."""
    repo_ids = repo_df['id'].tolist()

    # Filter to selected repos
    repo_pr_ids = pr_df[pr_df['repo_id'].isin(repo_ids)]['id'].tolist()
    repo_comments = comments_df[comments_df['pr_id'].isin(repo_pr_ids)]
    repo_reviews = reviews_df[reviews_df['pr_id'].isin(repo_pr_ids)]

    # Count activity per user
    comment_counts = repo_comments.groupby('user').size().reset_index(name='comment_count')
    review_counts = repo_reviews.groupby('user').size().reset_index(name='review_count')

    # Merge counts
    user_activity = comment_counts.merge(review_counts, on='user', how='outer').fillna(0)
    user_activity['total_activity'] = user_activity['comment_count'] + user_activity['review_count']

    # Get top N% most active users
    threshold = user_activity['total_activity'].quantile(1 - top_percent / 100)
    top_users = user_activity[user_activity['total_activity'] >= threshold].copy()

    # Identify bot patterns
    def detect_bot_pattern(username):
        patterns_found = []
        for pattern in BOT_PATTERNS:
            if re.search(pattern, str(username)):
                patterns_found.append(pattern)
        return patterns_found

    top_users['bot_patterns'] = top_users['user'].apply(detect_bot_pattern)
    top_users['is_bot'] = top_users['bot_patterns'].apply(lambda x: len(x) > 0)

    # Filter to only bot users
    bot_users = top_users[top_users['is_bot']].copy()

    # Calculate PR acceptance/rejection rates for PRs they reviewed
    results = []
    for _, bot in bot_users.iterrows():
        username = bot['user']

        # PRs reviewed by this user
        reviewed_pr_ids = repo_reviews[repo_reviews['user'] == username]['pr_id'].unique()
        reviewed_prs = pr_df[pr_df['id'].isin(reviewed_pr_ids)]

        total_reviewed = len(reviewed_prs)
        if total_reviewed > 0:
            accepted = reviewed_prs['accepted'].sum()
            rejected = reviewed_prs['rejected'].sum()
            acceptance_rate = accepted / total_reviewed * 100
            rejection_rate = rejected / total_reviewed * 100
        else:
            acceptance_rate = 0
            rejection_rate = 0

        results.append({
            'user': username,
            'patterns_detected': ', '.join(bot['bot_patterns']),
            'comment_count': int(bot['comment_count']),
            'review_count': int(bot['review_count']),
            'total_activity': int(bot['total_activity']),
            'prs_reviewed': total_reviewed,
            'acceptance_rate': round(acceptance_rate, 1),
            'rejection_rate': round(rejection_rate, 1)
        })

    return pd.DataFrame(results).sort_values('total_activity', ascending=False)


# ============ Visualization Functions ============

def create_pr_status_bar_chart(status_df: pd.DataFrame, output_dir: Path) -> None:
    """Create grouped bar chart for PR status per repository."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(status_df))
    width = 0.25

    ax.bar(x - width, status_df['accepted'], width, label='Accepted',
           color=OUTCOME_COLORS['accepted'], edgecolor='black', linewidth=0.5)
    ax.bar(x, status_df['rejected'], width, label='Rejected',
           color=OUTCOME_COLORS['rejected'], edgecolor='black', linewidth=0.5)
    ax.bar(x + width, status_df['open'], width, label='Open',
           color=OUTCOME_COLORS['open'], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Repository', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of PRs', fontsize=11, fontweight='bold')
    ax.set_title('PR Status by Repository', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)

    # Shorten repo names for display
    short_names = [name.split('/')[-1] if '/' in name else name for name in status_df['repo_name']]
    ax.set_xticklabels(short_names, rotation=45, ha='right')

    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (_, row) in enumerate(status_df.iterrows()):
        if row['accepted'] > 0:
            ax.text(i - width, row['accepted'] + 1, str(int(row['accepted'])),
                   ha='center', va='bottom', fontsize=8)
        if row['rejected'] > 0:
            ax.text(i, row['rejected'] + 1, str(int(row['rejected'])),
                   ha='center', va='bottom', fontsize=8)
        if row['open'] > 0:
            ax.text(i + width, row['open'] + 1, str(int(row['open'])),
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'pr_status_by_repo.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'pr_status_by_repo.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: pr_status_by_repo.png/pdf")


def create_turnaround_boxplot_grid(turnaround_data: dict, output_dir: Path) -> None:
    """Create boxplot grid for turnaround time (accepted vs rejected) per repository."""
    repo_names = list(turnaround_data.keys())
    n_repos = len(repo_names)
    n_cols = min(5, n_repos)
    n_rows = (n_repos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_repos == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, repo_name in enumerate(repo_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        data = turnaround_data[repo_name]
        data_to_plot = []
        labels = []
        colors = []

        if len(data['accepted']) > 0:
            data_to_plot.append(data['accepted'])
            labels.append('Accepted')
            colors.append(OUTCOME_COLORS['accepted'])
        if len(data['rejected']) > 0:
            data_to_plot.append(data['rejected'])
            labels.append('Rejected')
            colors.append(OUTCOME_COLORS['rejected'])

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        short_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
        ax.set_title(short_name, fontsize=10, fontweight='bold')
        ax.set_ylabel('Hours')
        ax.grid(axis='y', alpha=0.3)

    # Hide empty subplots
    for idx in range(n_repos, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('Turnaround Time by Repository (Accepted vs Rejected)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'turnaround_by_repo.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'turnaround_by_repo.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: turnaround_by_repo.png/pdf")


def create_churn_boxplot_grid(churn_data: dict, output_dir: Path) -> None:
    """Create boxplot grid for code churn metrics per repository."""
    repo_names = list(churn_data.keys())
    n_repos = len(repo_names)
    n_cols = min(5, n_repos)
    n_rows = (n_repos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_repos == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    metric_colors = ['#3498db', '#e74c3c', '#2ecc71']  # blue, red, green
    metric_labels = ['Added', 'Deleted', 'Files']

    for idx, repo_name in enumerate(repo_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        data = churn_data[repo_name]
        data_to_plot = [
            data['lines_added'],
            data['lines_deleted'],
            data['num_files_changed']
        ]

        # Filter out empty arrays
        valid_data = [(d, l, c) for d, l, c in zip(data_to_plot, metric_labels, metric_colors) if len(d) > 0]

        if valid_data:
            plot_data = [d[0] for d in valid_data]
            plot_labels = [d[1] for d in valid_data]
            plot_colors = [d[2] for d in valid_data]

            bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True, showfliers=False)
            for patch, color in zip(bp['boxes'], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        short_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
        ax.set_title(short_name, fontsize=10, fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)

    # Hide empty subplots
    for idx in range(n_repos, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('Code Changes by Repository (Lines Added/Deleted, Files Changed)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'churn_by_repo.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'churn_by_repo.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: churn_by_repo.png/pdf")


def create_task_type_bar_charts(task_type_df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar charts showing acceptance rate by task type per repository."""
    if task_type_df.empty:
        print("  No task type data available")
        return

    repo_names = task_type_df['repo_name'].unique()
    n_repos = len(repo_names)
    n_cols = min(5, n_repos)
    n_rows = (n_repos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_repos == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Color map for task types
    task_colors = plt.cm.tab10(np.linspace(0, 1, len(TASK_TYPE_ORDER)))
    task_color_map = dict(zip(TASK_TYPE_ORDER, task_colors))

    for idx, repo_name in enumerate(repo_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        repo_data = task_type_df[task_type_df['repo_name'] == repo_name]

        if len(repo_data) > 0:
            # Order by TASK_TYPE_ORDER
            repo_data = repo_data.set_index('task_type').reindex(TASK_TYPE_ORDER).dropna().reset_index()

            x = np.arange(len(repo_data))
            colors = [task_color_map.get(t, '#808080') for t in repo_data['task_type']]

            bars = ax.bar(x, repo_data['acceptance_rate'], color=colors, edgecolor='black', linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(repo_data['task_type'], rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 100)
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

        short_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
        ax.set_title(short_name, fontsize=10, fontweight='bold')
        ax.set_ylabel('Acceptance %')
        ax.grid(axis='y', alpha=0.3)

    # Hide empty subplots
    for idx in range(n_repos, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # Create legend handles for task types
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=task_color_map[t], edgecolor='black', linewidth=0.5, label=t)
                      for t in TASK_TYPE_ORDER]

    fig.suptitle('Acceptance Rate by Task Type per Repository', fontsize=14, fontweight='bold')
    fig.legend(handles=legend_handles, loc='lower center', ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_dir / 'task_type_by_repo.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'task_type_by_repo.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: task_type_by_repo.png/pdf")


def create_reviews_boxplot_grid(review_comment_data: dict, output_dir: Path) -> None:
    """Create boxplot grid for number of reviews per repository."""
    repo_names = list(review_comment_data.keys())
    n_repos = len(repo_names)
    n_cols = min(5, n_repos)
    n_rows = (n_repos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_repos == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, repo_name in enumerate(repo_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        data = review_comment_data[repo_name]
        data_to_plot = []
        labels = []
        colors = []

        if len(data['accepted_reviews']) > 0:
            data_to_plot.append(data['accepted_reviews'])
            labels.append('Accepted')
            colors.append(OUTCOME_COLORS['accepted'])
        if len(data['rejected_reviews']) > 0:
            data_to_plot.append(data['rejected_reviews'])
            labels.append('Rejected')
            colors.append(OUTCOME_COLORS['rejected'])

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        short_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
        ax.set_title(short_name, fontsize=10, fontweight='bold')
        ax.set_ylabel('# Reviews')
        ax.grid(axis='y', alpha=0.3)

    # Hide empty subplots
    for idx in range(n_repos, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('Number of Reviews per PR (Accepted vs Rejected)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'reviews_by_repo.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'reviews_by_repo.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: reviews_by_repo.png/pdf")


def create_comments_boxplot_grid(review_comment_data: dict, output_dir: Path) -> None:
    """Create boxplot grid for number of comments per repository."""
    repo_names = list(review_comment_data.keys())
    n_repos = len(repo_names)
    n_cols = min(5, n_repos)
    n_rows = (n_repos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_repos == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, repo_name in enumerate(repo_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        data = review_comment_data[repo_name]
        data_to_plot = []
        labels = []
        colors = []

        if len(data['accepted_comments']) > 0:
            data_to_plot.append(data['accepted_comments'])
            labels.append('Accepted')
            colors.append(OUTCOME_COLORS['accepted'])
        if len(data['rejected_comments']) > 0:
            data_to_plot.append(data['rejected_comments'])
            labels.append('Rejected')
            colors.append(OUTCOME_COLORS['rejected'])

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showfliers=False)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        short_name = repo_name.split('/')[-1] if '/' in repo_name else repo_name
        ax.set_title(short_name, fontsize=10, fontweight='bold')
        ax.set_ylabel('# Comments')
        ax.grid(axis='y', alpha=0.3)

    # Hide empty subplots
    for idx in range(n_repos, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle('Number of Comments per PR (Accepted vs Rejected)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'comments_by_repo.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comments_by_repo.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: comments_by_repo.png/pdf")


# ============ Report Generation ============

def generate_markdown_report(
    top_repos: pd.DataFrame,
    status_df: pd.DataFrame,
    agent_counts_df: pd.DataFrame,
    task_type_df: pd.DataFrame,
    bot_users_df: pd.DataFrame,
    output_dir: Path,
    args
) -> str:
    """Generate comprehensive markdown report."""
    star_col = 'stargazers_count' if 'stargazers_count' in top_repos.columns else 'stars'

    report = []
    report.append(f"# Multi-Repository Analysis Report")
    report.append("")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append(f"**Configuration:**")
    report.append(f"- Number of repositories analyzed: {args.num_repos}")
    report.append(f"- Minimum turnaround time: {args.min_turnaround} seconds")
    report.append(f"- Top outlier percentile excluded: {args.top_n_percent}%")
    report.append(f"- Bot detection threshold: Top {args.bot_threshold}% users")
    report.append("")

    # Top Repositories
    report.append("## Top Repositories by Stars")
    report.append("")
    report.append("| # | Repository | Stars | Language | License |")
    report.append("|---|------------|-------|----------|---------|")
    for idx, (_, repo) in enumerate(top_repos.iterrows(), 1):
        report.append(f"| {idx} | {repo['full_name']} | {repo[star_col]:,} | {repo.get('language', 'N/A')} | {repo.get('license', 'N/A')} |")
    report.append("")

    # PR Status Summary
    report.append("## PR Status Summary")
    report.append("")
    report.append("| Repository | Accepted | Rejected | Open | Total | Acceptance Rate |")
    report.append("|------------|----------|----------|------|-------|-----------------|")
    for _, row in status_df.iterrows():
        short_name = row['repo_name'].split('/')[-1]
        total = row['accepted'] + row['rejected'] + row['open']
        acc_rate = (row['accepted'] / total * 100) if total > 0 else 0
        report.append(f"| {short_name} | {row['accepted']} | {row['rejected']} | {row['open']} | {total} | {acc_rate:.1f}% |")
    report.append("")

    # Agent Distribution (Table 2)
    report.append("## Agent Distribution by Repository")
    report.append("")
    header = "| Repository |" + " | ".join(AGENTS) + " | Total |"
    separator = "|------------|" + " | ".join(["---"] * len(AGENTS)) + " | ----- |"
    report.append(header)
    report.append(separator)
    for _, row in agent_counts_df.iterrows():
        short_name = row['repo_name'].split('/')[-1]
        values = [str(int(row.get(agent, 0))) for agent in AGENTS]
        report.append(f"| {short_name} | " + " | ".join(values) + f" | {int(row['Total'])} |")
    report.append("")

    # Bot Users (Table 7)
    report.append("## Bot/AI Users Identified")
    report.append("")
    if len(bot_users_df) > 0:
        report.append(f"Found {len(bot_users_df)} bot/AI users among top {args.bot_threshold}% most active users:")
        report.append("")
        report.append("| User | Patterns | Comments | Reviews | PRs Reviewed | Acceptance % | Rejection % |")
        report.append("|------|----------|----------|---------|--------------|--------------|-------------|")
        for _, row in bot_users_df.iterrows():
            report.append(f"| {row['user']} | {row['patterns_detected']} | {row['comment_count']} | {row['review_count']} | {row['prs_reviewed']} | {row['acceptance_rate']}% | {row['rejection_rate']}% |")
    else:
        report.append("No bot/AI users identified in the selected repositories.")
    report.append("")

    # Visualizations
    report.append("## Visualizations")
    report.append("")
    report.append("The following visualizations have been generated in the `visualizations/` directory:")
    report.append("")
    report.append("1. **pr_status_by_repo.png/pdf** - Bar chart showing accepted, rejected, and open PRs per repository")
    report.append("2. **turnaround_by_repo.png/pdf** - Boxplot grid comparing turnaround times for accepted vs rejected PRs")
    report.append("3. **churn_by_repo.png/pdf** - Boxplot grid showing lines added, deleted, and files changed per repository")
    report.append("4. **task_type_by_repo.png/pdf** - Bar charts showing acceptance rate by task type per repository")
    report.append("5. **reviews_by_repo.png/pdf** - Boxplot grid comparing number of reviews for accepted vs rejected PRs")
    report.append("6. **comments_by_repo.png/pdf** - Boxplot grid comparing number of comments for accepted vs rejected PRs")
    report.append("")

    # Usage
    report.append("## How to Run")
    report.append("")
    report.append("```bash")
    report.append("# Default: analyze top 10 repositories")
    report.append("uv run specific_repo_exploration/multi_repo_analyzer.py")
    report.append("")
    report.append("# Analyze top N repositories")
    report.append("uv run specific_repo_exploration/multi_repo_analyzer.py --num-repos 15")
    report.append("")
    report.append("# Custom configuration")
    report.append("uv run specific_repo_exploration/multi_repo_analyzer.py -n 10 --min-turnaround 120 --top-n-percent 5")
    report.append("```")
    report.append("")

    report_text = "\n".join(report)

    # Save report
    report_path = output_dir / 'report.md'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"  Saved: report.md")
    return report_text


# ============ Main Function ============

def analyze_top_repos(args) -> None:
    """Main analysis function."""
    output_dir = Path(args.output)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Repository Analyzer for AIDev Dataset")
    print("=" * 60)
    print()

    # Load datasets
    datasets = load_datasets()
    pr_df = datasets['pr_df']
    repo_df = datasets['repo_df']

    print()

    # Get top N repos by stars
    top_repos = get_top_n_repos_by_stars(repo_df, pr_df, n=args.num_repos)
    top_repo_ids = top_repos['id'].tolist()

    print()

    # Filter PRs to selected repos
    filtered_pr_df = pr_df[pr_df['repo_id'].isin(top_repo_ids)].copy()
    print(f"PRs in selected repositories: {len(filtered_pr_df)}")

    # Annotate features
    print("\nAnnotating PR features...")
    annotated_pr_df = annotate_pr_features(filtered_pr_df, datasets['pr_commit_details_df'])

    # Apply filters
    print("\nApplying filters...")
    final_pr_df = apply_filters(
        annotated_pr_df,
        min_turnaround=args.min_turnaround,
        top_percent_lines=args.top_n_percent
    )
    print(f"Final PR count after filtering: {len(final_pr_df)}")

    print("\n" + "=" * 60)
    print("Computing Statistics")
    print("=" * 60)

    # Compute all statistics
    print("\nComputing PR status counts...")
    status_df = compute_pr_status_counts(final_pr_df, top_repos)

    print("Computing agent counts...")
    agent_counts_df = compute_agent_counts(final_pr_df, top_repos)

    print("Computing turnaround data...")
    turnaround_data = compute_turnaround_data(final_pr_df, top_repos)

    print("Computing churn data...")
    churn_data = compute_churn_data(final_pr_df, top_repos)

    print("Computing task type acceptance...")
    task_type_df = compute_task_type_acceptance(final_pr_df, datasets['pr_task_type_df'], top_repos)

    print("Computing review/comment data...")
    review_comment_data = compute_review_comment_data(
        final_pr_df,
        datasets['pr_reviews_df'],
        datasets['pr_comments_df'],
        top_repos
    )

    print("Identifying bot users...")
    bot_users_df = identify_bot_users(
        final_pr_df,
        datasets['pr_comments_df'],
        datasets['pr_reviews_df'],
        top_repos,
        top_percent=args.bot_threshold
    )

    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    print()

    # Create all visualizations
    create_pr_status_bar_chart(status_df, viz_dir)
    create_turnaround_boxplot_grid(turnaround_data, viz_dir)
    create_churn_boxplot_grid(churn_data, viz_dir)
    create_task_type_bar_charts(task_type_df, viz_dir)
    create_reviews_boxplot_grid(review_comment_data, viz_dir)
    create_comments_boxplot_grid(review_comment_data, viz_dir)

    print("\n" + "=" * 60)
    print("Generating Reports and Saving Data")
    print("=" * 60)
    print()

    # Save CSVs
    status_df.to_csv(output_dir / 'pr_status_counts.csv', index=False)
    print(f"  Saved: pr_status_counts.csv")

    agent_counts_df.to_csv(output_dir / 'agent_counts_by_repo.csv', index=False)
    print(f"  Saved: agent_counts_by_repo.csv")

    task_type_df.to_csv(output_dir / 'task_type_acceptance.csv', index=False)
    print(f"  Saved: task_type_acceptance.csv")

    bot_users_df.to_csv(output_dir / 'bot_users_analysis.csv', index=False)
    print(f"  Saved: bot_users_analysis.csv")

    star_col = 'stargazers_count' if 'stargazers_count' in top_repos.columns else 'stars'
    top_repos[['full_name', star_col, 'language', 'license']].to_csv(
        output_dir / 'top_repos_info.csv', index=False
    )
    print(f"  Saved: top_repos_info.csv")

    # Generate report
    generate_markdown_report(
        top_repos, status_df, agent_counts_df, task_type_df, bot_users_df,
        output_dir, args
    )

    # Save summary stats as JSON
    summary = {
        'num_repos_analyzed': args.num_repos,
        'total_prs_analyzed': len(final_pr_df),
        'filters_applied': {
            'min_turnaround_seconds': args.min_turnaround,
            'top_n_percent_lines_excluded': args.top_n_percent
        },
        'bot_users_found': len(bot_users_df),
        'repos': status_df.to_dict(orient='records')
    }

    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: summary_stats.json")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Visualizations saved to: {viz_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-repository PR analysis for top N repos by stars in AIDev dataset"
    )

    parser.add_argument(
        "--num-repos", "-n",
        type=int,
        default=10,
        help="Number of top repositories to analyze (default: 10)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./specific_repo_exploration/multi_repo_analysis",
        help="Output directory for results"
    )

    parser.add_argument(
        "--min-turnaround",
        type=int,
        default=60,
        help="Minimum turnaround time in seconds to include PR (default: 60)"
    )

    parser.add_argument(
        "--top-n-percent",
        type=int,
        default=10,
        help="Filter out PRs in top N%% for lines modified (default: 10)"
    )

    parser.add_argument(
        "--bot-threshold",
        type=float,
        default=15.0,
        help="Top %% of users to consider for bot detection (default: 15)"
    )

    args = parser.parse_args()
    analyze_top_repos(args)


if __name__ == "__main__":
    main()
