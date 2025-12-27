import pandas as pd
import scipy.stats as stats
import numpy as np


def conditional_acceptance_rate(df, bool_col):
    accepted_with_condition = df[df[bool_col]]['accepted'].sum()
    total_with_condition = df[df[bool_col]].shape[0]
    rate_with_condition = accepted_with_condition / \
        total_with_condition if total_with_condition > 0 else 0

    accepted_without_condition = df[~df[bool_col]]['accepted'].sum()
    total_without_condition = df[~df[bool_col]].shape[0]
    rate_without_condition = accepted_without_condition / \
        total_without_condition if total_without_condition > 0 else 0

    print(
        f"Acceptance rate for PRs with {bool_col}: {rate_with_condition * 100} ({accepted_with_condition}/{total_with_condition})")
    print(
        f"Acceptance rate for PRs without {bool_col}: {rate_without_condition * 100} ({accepted_without_condition}/{total_without_condition})")


def df_column_statistics(pr_df, column_name):
    """
    Analyse statistics for a given column split by accepted and rejected PRs.

    Parameters:
    - pr_df: DataFrame containing PR data
    - column_name: Name of the column to analyze

    Returns:
    - Dictionary containing statistics for accepted and rejected PRs
    """
    # Split sets into accepted and rejected PRs
    accepted_prs = pr_df[pr_df['accepted'] == True]
    rejected_prs = pr_df[pr_df['rejected'] == True]

    # Calculate statistics for accepted PRs
    mean_accepted = accepted_prs[column_name].mean()
    median_accepted = accepted_prs[column_name].median()
    std_accepted = accepted_prs[column_name].std()
    min_accepted = accepted_prs[column_name].min()
    max_accepted = accepted_prs[column_name].max()

    # Calculate statistics for rejected PRs
    mean_rejected = rejected_prs[column_name].mean()
    median_rejected = rejected_prs[column_name].median()
    std_rejected = rejected_prs[column_name].std()
    min_rejected = rejected_prs[column_name].min()
    max_rejected = rejected_prs[column_name].max()

    # Print results
    print(f"Accepted PR Statistics for '{column_name}':")
    print(f"Mean = {mean_accepted}")
    print(f"Median = {median_accepted}")
    print(f"Standard Deviation = {std_accepted}")
    print(f"Min = {min_accepted}")
    print(f"Max = {max_accepted}")

    print(f"\nRejected PR Statistics for '{column_name}':")
    print(f"Mean = {mean_rejected}")
    print(f"Median = {median_rejected}")
    print(f"Standard Deviation = {std_rejected}")
    print(f"Min = {min_rejected}")
    print(f"Max = {max_rejected}")


def fishers_exact_test(pr_df, feature_column):
    """
    Perform Fisher's Exact Test to analyze the relationship between a feature and PR acceptance.

    Parameters:
    - pr_df: DataFrame containing PR data
    - feature_column: Name of the boolean column to test (e.g., 'related_issue')

    Returns:
    - Dictionary containing test results
    """
    # Create contingency table
    contingency_table = pd.crosstab(pr_df[feature_column], pr_df['accepted'])

    print(f"Fisher's Exact Test for '{feature_column}' vs 'accepted'")
    print("\nContingency Table:")
    print(contingency_table)

    # Perform Fisher's Exact Test
    oddsratio, p_value = stats.fisher_exact(contingency_table)

    print(f"\nOdds Ratio: {oddsratio:.4f}")
    print(f"P-value: {p_value:.4f}")


def chi_squared_test(pr_df, feature_column):
    """
    Perform Chi-squared Test to analyze the relationship between a feature and PR acceptance,
    and report effect size (Cramér's V).

    Parameters:
    - pr_df: DataFrame containing PR data
    - feature_column: Name of the boolean/categorical column to test (e.g., 'has_review')
    """
    # Contingency table
    contingency_table = pd.crosstab(pr_df[feature_column], pr_df["accepted"])

    print(f"Chi-squared Test for '{feature_column}' vs 'accepted'")
    print("\nContingency Table:")
    print(contingency_table)

    # Chi-squared test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Sample size N (sum of all cells)
    n = contingency_table.to_numpy().sum()

    # Cramér's V
    r, c = contingency_table.shape
    k = min(r, c)
    if k <= 1 or n == 0:
        cramer_v = np.nan
    else:
        cramer_v = np.sqrt(chi2 / (n * (k - 1)))

    print(f"\nChi-squared statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.15f}")
    print(f"Degrees of freedom: {dof}")
    print(f"N: {n}")
    print(f"Cramér's V: {cramer_v:.4f}")


def filter_top_n_for_cols(pr_dataframe: pd.DataFrame, col_list: list[str], filter_percent=10):
    def top_n_percent_ids(pr_dataframe, col_name, percent):
        threshold = pr_dataframe[col_name].quantile((100 - percent) / 100)
        top_n_percent = pr_dataframe[pr_dataframe[col_name] >= threshold]
        return set(top_n_percent['id'].tolist())

    exclusion_sets = []
    for col in col_list:
        col_exclusion_ids = top_n_percent_ids(
            pr_dataframe, col, filter_percent)
        print(f"PRs to exclude for {col}: {len(col_exclusion_ids)}")
        exclusion_sets.append(col_exclusion_ids)

    full_exclusion_list = set.union(*exclusion_sets)
    print(f"Total Rows to Filter: {len(full_exclusion_list)}")
    return pr_dataframe[~pr_dataframe['id'].isin(full_exclusion_list)]
