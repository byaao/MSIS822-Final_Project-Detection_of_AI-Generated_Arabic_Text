
"""visualization.py — Functions for generating plots and visualizations (Arabic-friendly)

This module provides:
  - wordcloud_for: Generate an Arabic-friendly word cloud from a DataFrame column.
  - _top_ngrams:   Compute top n-grams using scikit-learn's CountVectorizer.
  - compute_by_groups: Compute n-grams grouped by one or more columns.
  - plot_ngrams:   Plot grouped n-gram frequency bar charts (with Arabic shaping).

Notes
-----
* Arabic text rendering in Matplotlib can be tricky. We use arabic_reshaper and python-bidi
  to display text correctly from right-to-left. If these packages are not installed,
  the functions will gracefully fall back to displaying raw text.
* The class `ArabicWordCloud` is expected to be available from the `arabicwordcloud` package.
  If unavailable, an ImportError will be raised when calling `wordcloud_for`.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import Arabic shapers; fall back to no-op if missing
try:
    import arabic_reshaper
    from bidi.algorithm import get_display as _bidi_get_display
except Exception:  # pragma: no cover
    arabic_reshaper = None
    _bidi_get_display = None

# Word cloud (ArabicWordCloud)
try:
    from arabicwordcloud import ArabicWordCloud
except Exception as e:  # Module may not be installed at import time; raise at call site if needed
    ArabicWordCloud = None  # type: ignore

# N-gram vectorizer
try:
    from sklearn.feature_extraction.text import CountVectorizer
except Exception as e:  # pragma: no cover
    CountVectorizer = None  # type: ignore


def ar_shape(text: str) -> str:
    """Return Arabic-shaped, bidi-corrected text for plotting.

    If the necessary libraries aren't available, return the text unchanged.
    """
    if text is None:
        return ""
    if arabic_reshaper is None or _bidi_get_display is None:
        return str(text)
    try:
        reshaped = arabic_reshaper.reshape(str(text))
        return _bidi_get_display(reshaped)
    except Exception:
        return str(text)


def wordcloud_for(sub_df: pd.DataFrame, title: str, *, max_docs: Optional[int] = None, ax=None):
    """Generate a word cloud for a given sub-dataframe.

    Parameters
    ----------
    sub_df : DataFrame
        Must contain a column named "text_clean" with pre-tokenized (space-separated) text.
    title : str
        Title to show over the figure/axes.
    max_docs : int, optional
        If provided and the subset is larger, sample up to this many rows (unbiased) for speed.
    ax : matplotlib axis, optional
        If given, draw on this axis; otherwise create a new figure.
    """
    if ArabicWordCloud is None:
        raise ImportError("arabicwordcloud is required for `wordcloud_for`. Install `arabicwordcloud`. ")

    if "text_clean" not in sub_df.columns:
        raise KeyError("`sub_df` must contain a 'text_clean' column.")

    s = sub_df["text_clean"].astype(str)

    # Optional unbiased downsampling
    if max_docs is not None and len(s) > max_docs:
        s = s.sample(n=max_docs, random_state=42)

    # Token frequency
    freq = Counter()
    for doc in s:
        freq.update(doc.split())

    wc = ArabicWordCloud(
        width=900,
        height=500,
        background_color="white",
        collocations=False,
    ).generate_from_frequencies(freq)

    if ax is None:
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=16)
        plt.show()
    else:
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=12)


def _top_ngrams(text: Iterable[str], n: int = 1, top_k: int = 20, min_df: Union[int, float] = 2, max_df: Union[int, float] = 1.0) -> pd.DataFrame:
    """Return top n-grams (by count).

    Parameters
    ----------
    text : iterable of str
        The corpus (each item is a document/string). Will be cast to string.
    n : int
        N-gram length (1 = unigrams, 2 = bigrams, etc.).
    top_k : int
        Number of top items to return.
    min_df, max_df : int or float
        scikit-learn CountVectorizer thresholds; see its documentation.

    Returns
    -------
    DataFrame with columns [ngram, freq] sorted by descending freq.
    """
    if CountVectorizer is None:
        raise ImportError("scikit-learn is required for `_top_ngrams`. Install `scikit-learn`. ")

    # Ensure we have a pandas Series of strings
    s = pd.Series(list(text), dtype="string").fillna("").astype(str)

    vec = CountVectorizer(
        analyzer="word",
        ngram_range=(n, n),
        token_pattern=r"(?u)\b[^\W\d_]+\b",  # letters only (Arabic OK)
        min_df=min_df,
        max_df=max_df,
        max_features=50000,
    )
    X = vec.fit_transform(s.astype(str))
    # Sum over rows to get feature counts
    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab  = vec.get_feature_names_out()
    out = (
        pd.DataFrame({"ngram": vocab, "freq": freqs})
          .nlargest(top_k, "freq")
          .reset_index(drop=True)
    )
    return out


def compute_by_groups(df: pd.DataFrame, group_cols: Union[str, Sequence[str]], *, text_col: str = "text_clean", n_list: Sequence[int] = (1, 2, 3),
                      top_k: int = 20, min_df: Union[int, float] = 2, max_df: Union[int, float] = 1.0) -> Dict[Tuple, Dict[int, pd.DataFrame]]:
    """Compute top n-grams for groups in a DataFrame.

    Returns
    -------
    dict mapping group_key (tuple) -> {n: DataFrame(ngram,freq)}
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if text_col not in df.columns:
        raise KeyError(f"`df` must contain a '{text_col}' column.")

    results: Dict[Tuple, Dict[int, pd.DataFrame]] = {}
    for key, sub in df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        s = sub[text_col].astype(str)
        results[key] = {n: _top_ngrams(s, n=n, top_k=top_k, min_df=min_df, max_df=max_df)
                        for n in n_list}
    return results


def plot_ngrams(rows: Sequence, n_list: Sequence[int], results: Mapping[Tuple, Mapping[int, pd.DataFrame]],
                filter_fn: Callable[[Tuple], bool], *,
                row_label_fmt: Callable[[object], str] = str, title: str = "", figsize_cell: Tuple[float, float] = (5, 4), dpi: int = 150,
                row_dim: int = 0):
    """Plot a grid of horizontal bar charts for grouped n-grams.

    Parameters
    ----------
    rows : sequence
        The ordered list of row keys (values drawn from position `row_dim` in a group key tuple).
    n_list : sequence[int]
        The n values to plot per column (e.g., (1,2,3) for uni/bi/tri-grams).
    results : mapping
        Mapping: group_key(tuple) -> { n(int): DataFrame with columns ['ngram','freq'] }.
    filter_fn : callable
        Function that takes a group key and returns True if it should be included.
    row_label_fmt : callable, default=str
        Formatter for the row labels displayed on the left.
    title : str
        Figure title.
    figsize_cell : (w,h)
        Size per cell (per subplot).
    dpi : int
        Figure DPI.
    row_dim : int
        Which position in the group key corresponds to the row.
    """
    n_rows, n_cols = len(rows), len(n_list)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_cell[0]*n_cols, figsize_cell[1]*n_rows),
                             dpi=dpi, constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    # shared x-limits per n (column)
    xlims = {}
    for n in n_list:
        mx = 0
        for k in results:
            if filter_fn(k) and not results[k][n].empty:
                mx = max(mx, results[k][n]["freq"].max())
        xlims[n] = mx * 1.05 if mx > 0 else 1

    # column headers
    for j, n in enumerate(n_list):
        axes[0][j].set_title(f"{n}-gram", fontsize=14, pad=10)

    # plot grid
    for i, row_val in enumerate(rows):
        for j, n in enumerate(n_list):
            ax = axes[i][j]

            # find subset for this row & n
            match = None
            for k in results.keys():
                if filter_fn(k) and k[row_dim] == row_val:
                    match = results[k][n]
                    break

            if match is None or match.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, ar_shape("لا توجد بيانات"), ha="center", va="center", fontsize=11)
                continue

            y_labels = match["ngram"].apply(ar_shape).iloc[::-1]
            x_vals   = match["freq"].iloc[::-1]
            bars = ax.barh(y_labels, x_vals)

            # annotate
            for b, v in zip(bars, x_vals):
                ax.text(v + xlims[n]*0.01, b.get_y()+b.get_height()/2,
                        f"{int(v)}", va="center", fontsize=9)

            ax.set_xlim(0, xlims[n])
            ax.set_ylabel("")
            ax.set_xlabel(ar_shape("التكرار") + " / Frequency", fontsize=11)
            if j == 0:
                ax.text(-0.02, 1.02, ar_shape(row_label_fmt(row_val)),
                        transform=ax.transAxes, fontsize=12, fontweight="bold",
                        ha="right", va="bottom")

    if title:
        fig.suptitle(title, fontsize=18, y=1.02)
    plt.show()


__all__ = [
    "ar_shape",
    "wordcloud_for",
    "_top_ngrams",
    "compute_by_groups",
    "plot_ngrams",
]
