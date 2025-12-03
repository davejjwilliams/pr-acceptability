# Understanding Factors Affecting PR Acceptability

Dataset:

- https://huggingface.co/datasets/hao-li/AIDev
- https://github.com/SAILResearch/AI_Teammates_in_SE3


## TODOs

- [ ] Explore which columns/tables are relevant from the dataset and their feasibility for analysis
    - [ ] Pre Factors
    - [ ] During Factors
    - [ ] Post Factors
- [ ] Explore initial correlations between acceptance and the factors identified in the Google Slide above:
    - [ ] Pre Factors
    - [ ] During Factors
    - [ ] Post Factors
- [ ] Explore techniques for modelling combinations of factors together

## Repository Structure

- `initial-exploration/`: Directories containing initial analysis of factors which may affect PR acceptability.
    - `pre-factors/`
    - `during-factors/`
    - `post-factors/`
- `original-analysis-scripts/`: Analysis scripts from dataset repository (linked above)
- `requirements.txt`: Requirements taken from dataset repository (linked above), feel free to add more

## Setup

1. Create a virtual environment.

```bash
python -m venv venv
```

2. Load virtual environment.

Linux/MacOS:
```bash
source venv/bin/activate
```

Windows:
```bash
venv/Scripts/activate
```

3. Install dependencies.
```bash
pip install -r requirements.txt
```

# Data Pre-Processing

Before continuing to analysis scripts, run the cells in the [`data_prep.ipynb`](./data_prep.ipynb) notebook to annotate the PR dataframe with additional columns (e.g. turnaround time, code churn, etc.).

The resulting dataframes can be found as CSVs under [`./data/filtered/`](./data/filtered/), which will be used in subsequent analysis scripts.

A copy of the original dataset will also be saved under [`./data/original/`](./data/original/).