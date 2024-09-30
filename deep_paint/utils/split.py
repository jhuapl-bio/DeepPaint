"""Split metadata into stratified train/val/test splits."""
import logging
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


def split(
    metadata: pd.DataFrame,
    stratify_col: str,
    group_col: Optional[str] = None,
    sizes: Optional[List[float]] = None,
    random_state: int = 0
) -> List:
    """
    Split metadata into stratified train/val/test splits.

    Metadata requires the 'stratify_col' column. If the 'group_col' column is 
    specified, then the splits will be stratified by the 'stratify' column 
    within the 'group' column. Only train/val splits are supported for 
    grouped stratified splitting, therefore the last element of `sizes` is only 
    used if 'group_col' is None.

    Parameters
    ----------
    metadata: pandas.DataFrame
        image metadata
    stratify_col: str
        column name of metadata on which to stratify split
    group_col: str, optional
        column name of metadata on which to group by
    sizes: List[float], optional
        sizes of train, val, and test splits
    random_state: int, default 0
        random seed to ensure reproducibility of split
    """
    assert stratify_col in metadata.columns, \
        f"Stratify column: `{stratify_col}` not in metadata."
    # Add index column to allow merging of predictions
    if "index" not in metadata.columns:
        metadata["index"] = metadata.index
    if group_col is None:
        logging.info("Stratified Split")
        if sizes is None:
            raise ValueError("Sizes must be specified for stratified split.")
        return _stratified_split(
            metadata=metadata,
            stratify_col=stratify_col,
            sizes=sizes,
            random_state=random_state
        )
    else:
        assert group_col in metadata.columns, \
        f"Group by column: `{group_col}` not in metadata."
        logging.info("Stratified Group Split")
        if sizes is None:
            raise ValueError("Sizes must be specified for stratified group split.")
        return _stratified_group_split(
            metadata=metadata,
            stratify_col=stratify_col,
            group_col=group_col,
            sizes=sizes
        )


def split_embeddings(
    df: pd.DataFrame
)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split logged embeddings into three pandas.DataFrames.
    
    This is necessary to utilize the `plot_confusion_matrix` and `plot_pca`
    functions outside of a model training run.
    """
    # get idx of raw embeddings
    last_num_idx = None
    for i, col in enumerate(df.columns):
        try:
            int(col)
            last_num_idx = i
        except ValueError:
            break
    embeddings = df[df.columns[:last_num_idx + 1]]
    # get predictions
    preds_cols = [col for col in df.columns if col.startswith("y")]
    predictions = df[preds_cols + ['index']]
    # get metadata
    metadata_cols = df.columns[last_num_idx + 1:]
    metadata_cols = [col for col in metadata_cols if not col.startswith("y")]
    metadata = df[metadata_cols]
    return embeddings, predictions, metadata


def _stratified_split(
    metadata: pd.DataFrame,
    stratify_col: str,
    sizes: List[float],
    random_state: int = 42
) -> List[pd.DataFrame]:
    """
    Wrapper for sklearn's `train_test_split` function to get stratified splits.

    Parameters
    ----------
    metadata: pandas.DataFrame
        image metadata
    stratify_col: str
        column name of metadata on which to stratify split
    sizes: List[float]
        sizes of train, val, and test splits
    random_state: int, default 42
        random seed to ensure reproducibility of split
    """
    assert sum(sizes) == 1, "Sizes must sum to 1."
    train_size, val_size, test_size = sizes
    # First split (train/val_test)
    metadata_train, metadata_val_test = train_test_split(
        metadata,
        train_size=train_size,
        stratify=metadata[stratify_col],
        random_state=random_state
    )
    if test_size == 0.0:
        return [
            metadata_train.reset_index(drop=True),
            metadata_val_test.reset_index(drop=True)
        ]
    else:
        # Second split (val/test)
        val_test_size = 1 - train_size
        val_size_adjusted = val_size / val_test_size
        metadata_val, metadata_test = train_test_split(
            metadata_val_test,
            train_size=val_size_adjusted,
            stratify=metadata_val_test[stratify_col],
            random_state=random_state
        )
        return [
            metadata_train.reset_index(drop=True),
            metadata_val.reset_index(drop=True),
            metadata_test.reset_index(drop=True)
        ]


def _stratified_group_split(
    metadata: pd.DataFrame,
    stratify_col: str,
    group_col: str,
    sizes: List[float]
):
    """
    Split metadata into stratified train/val/ splits grouped by a column.

    Wrapper for sklearn's `StratifiedGroupKFold` to get stratified group splits
    and takes the first fold as the train/val split.

    Parameters
    ----------
    metadata: pandas.DataFrame
        image metadata
    stratify_col: str
        column name of metadata on which to stratify split
    group_col: str
        column name of metadata on which to group by
    sizes: List[float]
        sizes of train, val, and test splits
    """
    assert sum(sizes) == 1, "Sizes must sum to 1."
    _, val_size, test_size = sizes
    if test_size != 0:
        logging.warning("Test size should be 0, ignoring test size.")
    n_splits=int(1/val_size)

    group_kfold = StratifiedGroupKFold(n_splits=n_splits)
    split_gen = group_kfold.split(metadata, metadata[stratify_col], groups=metadata[group_col])
    train_idx, val_idx = next(split_gen)
    train_fold = metadata.iloc[train_idx].reset_index(drop=True)
    val_fold = metadata.iloc[val_idx].reset_index(drop=True)
    return [train_fold, val_fold]
