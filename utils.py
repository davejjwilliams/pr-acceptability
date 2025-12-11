import pandas as pd

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