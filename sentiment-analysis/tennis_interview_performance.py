import argparse
import json
import re
import time
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.stats import pointbiserialr
from transformers import pipeline
from wordcloud import WordCloud


# ============================================================
# Configuration
# ============================================================

ORIGINAL_DATASET = "data/processed/player_tournament_interview_dataset.csv"
SENTIMENT_DATASET = "sentiment-analysis/interviews_with_sentiment.csv"

INTERVIEW_COLUMN = "first_pre_match_interview_qa_json"
PLAYER_COLUMN = "player_name"
RANK_COLUMN = "player_rank"
SENTIMENT_COLUMN = "sentiment_score"
PERFORMANCE_COLUMN = "current_finish_at_least_recent_average"

SENTIMENT_MODEL_NAME = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

MIN_TOURNAMENTS = 5
NUM_PLAYERS_PER_GROUP = 5


# ============================================================
# Data loading
# ============================================================

def load_data(csv_path):
    """Load the tennis interview dataset from a CSV file."""
    return pd.read_csv(csv_path)


# ============================================================
# Text preprocessing
# ============================================================

def load_stop_words():
    """Load English NLTK stopwords, downloading them if necessary."""
    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        words = stopwords.words("english")

    return set(words)


def parse_interview_json(qa_json):
    """Parse one interview JSON value safely."""
    if pd.isna(qa_json):
        return {}

    try:
        return json.loads(qa_json)
    except (TypeError, json.JSONDecodeError):
        return {}


def extract_answer_list(qa_json):
    """Return the player's non-empty answers from one interview."""
    qa = parse_interview_json(qa_json)
    answers = []

    for key, value in qa.items():
        if key.startswith("answer_") and isinstance(value, str):
            text = value.strip()

            if text:
                answers.append(text)

    return answers


def get_clean_words(qa_json, stop_words):
    """Extract cleaned English words from the player's answers."""
    clean_words = []

    for answer in extract_answer_list(qa_json):
        words = re.findall(r"\b[a-z']+\b", answer.lower())

        clean_words.extend(
            word
            for word in words
            if word not in stop_words and len(word) > 1
        )

    return clean_words


def collect_words(df, stop_words, player_names=None):
    """Collect cleaned words from all interviews, optionally for selected players."""
    words = []

    for _, row in df.iterrows():
        if player_names is not None and row[PLAYER_COLUMN] not in player_names:
            continue

        words.extend(
            get_clean_words(
                row[INTERVIEW_COLUMN],
                stop_words,
            )
        )

    return words


# ============================================================
# General word-frequency analysis
# ============================================================

def print_word_statistics(words, top_n=20):
    """Print total, unique, and most common words."""
    word_freq = Counter(words)

    print("Total words:", len(words))
    print("Unique words:", len(word_freq))
    print(f"\n{top_n} most common words:")

    for word, count in word_freq.most_common(top_n):
        print(word, count)

    return word_freq


def plot_wordcloud(word_freq, title):
    """Display a word cloud from a frequency dictionary."""
    if not word_freq:
        print(f"Skipping '{title}': no words were found.")
        return

    wordcloud = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        max_words=100,
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(14, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def run_general_word_analysis(df):
    """Analyze the most frequent words across all interviews."""
    stop_words = load_stop_words()

    all_words = collect_words(df, stop_words)
    all_word_freq = print_word_statistics(all_words)

    plot_wordcloud(
        all_word_freq,
        "Most Frequent Words in Pre-Match Interviews",
    )


# ============================================================
# Ranked-player word analysis
# ============================================================

def select_ranked_player_groups(
    df,
    min_tournaments=MIN_TOURNAMENTS,
    num_players=NUM_PLAYERS_PER_GROUP,
):
    """Select higher- and lower-ranked players using average valid rank."""

    rank_df = df[
        df[RANK_COLUMN].notna()
        & (df[RANK_COLUMN] != -1)
    ].copy()

    player_rank_stats = (
        rank_df.groupby(PLAYER_COLUMN)
        .agg(
            avg_player_rank=(RANK_COLUMN, "mean"),
            num_tournaments=(RANK_COLUMN, "count"),
        )
        .reset_index()
    )

    eligible_players = player_rank_stats[
        player_rank_stats["num_tournaments"] >= min_tournaments
    ].copy()

    top_players = eligible_players.nsmallest(
        num_players,
        "avg_player_rank",
    )

    bottom_players = eligible_players.nlargest(
        num_players,
        "avg_player_rank",
    )

    return top_players, bottom_players


def print_selected_players(top_players, bottom_players):
    """Print the selected higher- and lower-ranked player tables."""

    print("\nTOP 5 HIGHER-RANKED PLAYERS:")
    print(top_players.to_string(index=False))

    print("\nBOTTOM 5 LOWER-RANKED PLAYERS:")
    print(bottom_players.to_string(index=False))


def get_group_word_frequencies(
    df,
    top_players,
    bottom_players,
    stop_words,
):
    """Collect and count words for higher- and lower-ranked player groups."""

    top_names = set(top_players[PLAYER_COLUMN])
    bottom_names = set(bottom_players[PLAYER_COLUMN])

    top_words = collect_words(
        df,
        stop_words,
        top_names,
    )

    bottom_words = collect_words(
        df,
        stop_words,
        bottom_names,
    )

    return (
        top_words,
        bottom_words,
        Counter(top_words),
        Counter(bottom_words),
    )


def plot_group_wordclouds(
    top_word_freq,
    bottom_word_freq,
):
    """Display word clouds for higher- and lower-ranked players."""

    if not top_word_freq or not bottom_word_freq:
        print(
            "Skipping group word clouds: "
            "one of the groups has no words."
        )
        return

    top_wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        max_words=100,
    ).generate_from_frequencies(top_word_freq)

    bottom_wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        max_words=100,
    ).generate_from_frequencies(bottom_word_freq)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 7),
    )

    axes[0].imshow(
        top_wordcloud,
        interpolation="bilinear",
    )
    axes[0].axis("off")
    axes[0].set_title(
        "Higher-Ranked Players",
        fontsize=16,
    )

    axes[1].imshow(
        bottom_wordcloud,
        interpolation="bilinear",
    )
    axes[1].axis("off")
    axes[1].set_title(
        "Lower-Ranked Players",
        fontsize=16,
    )

    plt.suptitle(
        "Word Usage: Higher-Ranked vs. Lower-Ranked Players",
        fontsize=19,
    )

    plt.tight_layout()
    plt.show()


def plot_most_common_words(
    top_words,
    bottom_words,
    top_n=10,
):
    """Compare relative frequencies of the most common words in both groups."""

    if not top_words or not bottom_words:
        print(
            "Skipping frequency plots: "
            "one of the groups has no words."
        )
        return

    top_word_freq = Counter(top_words)
    bottom_word_freq = Counter(bottom_words)

    top_common = [
        (word, count / len(top_words))
        for word, count in top_word_freq.most_common(top_n)
    ]

    bottom_common = [
        (word, count / len(bottom_words))
        for word, count in bottom_word_freq.most_common(top_n)
    ]

    max_freq = max(
        max(freq for _, freq in top_common),
        max(freq for _, freq in bottom_common),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    top_words_plot, top_freq_plot = zip(*top_common)

    axes[0].barh(
        top_words_plot[::-1],
        top_freq_plot[::-1],
    )

    axes[0].set_title(
        "Higher-Ranked Players"
    )
    axes[0].set_xlabel(
        "Relative Frequency"
    )
    axes[0].set_xlim(
        0,
        max_freq * 1.1,
    )

    bottom_words_plot, bottom_freq_plot = zip(
        *bottom_common
    )

    axes[1].barh(
        bottom_words_plot[::-1],
        bottom_freq_plot[::-1],
    )

    axes[1].set_title(
        "Lower-Ranked Players"
    )
    axes[1].set_xlabel(
        "Relative Frequency"
    )
    axes[1].set_xlim(
        0,
        max_freq * 1.1,
    )

    plt.suptitle(
        "Most Frequent Words by Player Ranking Group",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


def run_ranked_word_analysis(df):
    """Compare word usage between higher- and lower-ranked players."""

    stop_words = load_stop_words()

    top_players, bottom_players = (
        select_ranked_player_groups(df)
    )

    print_selected_players(
        top_players,
        bottom_players,
    )

    (
        top_words,
        bottom_words,
        top_word_freq,
        bottom_word_freq,
    ) = get_group_word_frequencies(
        df,
        top_players,
        bottom_players,
        stop_words,
    )

    plot_group_wordclouds(
        top_word_freq,
        bottom_word_freq,
    )

    plot_most_common_words(
        top_words,
        bottom_words,
    )


# ============================================================
# Sentiment model
# ============================================================

def load_sentiment_model():
    """Load the Hugging Face sentiment-analysis pipeline."""

    print("Loading sentiment model...")

    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_NAME,
    )


def get_answer_sentiment(
    answer,
    sentiment_model,
):
    """Return one answer's sentiment on a [-1, 1] scale."""

    try:
        result = sentiment_model(
            answer,
            truncation=True,
        )[0]

        score = result["score"]

        if result["label"] == "POSITIVE":
            return score

        return -score

    except Exception:
        return np.nan


def get_interview_sentiment(
    qa_json,
    sentiment_model,
):
    """Return the mean answer-level sentiment for one interview."""

    answers = extract_answer_list(qa_json)

    if not answers:
        return np.nan

    scores = [
        get_answer_sentiment(
            answer,
            sentiment_model,
        )
        for answer in answers
    ]

    scores = [
        score
        for score in scores
        if not np.isnan(score)
    ]

    if not scores:
        return np.nan

    return float(np.mean(scores))


def run_sentiment_analysis(
    df,
    sentiment_model,
):
    """Compute interview-level sentiment for the full dataframe."""

    sentiment_results = []

    total = len(df)
    start_time = time.time()

    print(
        f"\nStarting sentiment analysis "
        f"on {total} interviews...\n"
    )

    for i, qa_json in enumerate(
        df[INTERVIEW_COLUMN]
    ):
        score = get_interview_sentiment(
            qa_json,
            sentiment_model,
        )

        sentiment_results.append(score)

        if (
            (i + 1) % 10 == 0
            or (i + 1) == total
        ):
            elapsed = (
                time.time()
                - start_time
            )

            avg_time = (
                elapsed
                / (i + 1)
            )

            remaining = (
                avg_time
                * (total - (i + 1))
            )

            percent = (
                ((i + 1) / total)
                * 100
            )

            print(
                f"{i + 1}/{total} "
                f"({percent:.1f}%) | "
                f"Elapsed: "
                f"{elapsed / 60:.1f} min | "
                f"Estimated remaining: "
                f"{remaining / 60:.1f} min",
                flush=True,
            )

    result_df = df.copy()

    result_df[
        SENTIMENT_COLUMN
    ] = sentiment_results

    total_time = (
        time.time()
        - start_time
    )

    print(
        "\n=========================================="
    )
    print("DONE")
    print(
        f"Total time: "
        f"{total_time / 60:.1f} minutes"
    )
    print(
        "=========================================="
    )

    return result_df


# ============================================================
# Sentiment sanity check
# ============================================================

def sentiment_sanity_check(
    sentiment_model,
):
    """Print one positive and one negative model test."""

    print("Positive test:")

    print(
        get_answer_sentiment(
            "I feel like I'm playing great tennis this year.",
            sentiment_model,
        )
    )

    print("\nNegative test:")

    print(
        get_answer_sentiment(
            "I was very sad about losing another final.",
            sentiment_model,
        )
    )


def run_sanity_check():
    """Load the model and run the sentiment sanity check."""

    sentiment_model = (
        load_sentiment_model()
    )

    sentiment_sanity_check(
        sentiment_model
    )


# ============================================================
# Run sentiment model on all interviews
# ============================================================

def run_model_on_interviews():
    """
    Run the sentiment model on all interviews.

    The resulting dataframe is kept in memory only.
    The provided precomputed sentiment CSV is not changed.
    """

    df = load_data(
        ORIGINAL_DATASET
    )

    sentiment_model = (
        load_sentiment_model()
    )

    result_df = (
        run_sentiment_analysis(
            df,
            sentiment_model,
        )
    )

    print(
        "\nSentiment analysis completed."
    )

    print(
        "The results were not saved "
        "to a CSV file."
    )

    return result_df


# ============================================================
# Sentiment plots and statistical analysis
# ============================================================

def plot_sentiment_distribution(df):
    """Plot the distribution of interview sentiment scores."""

    sentiment_data = (
        df[SENTIMENT_COLUMN]
        .dropna()
    )

    if sentiment_data.empty:
        print(
            "Skipping sentiment distribution: "
            "no sentiment scores found."
        )
        return

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        sentiment_data,
        bins=20,
        edgecolor="black",
    )

    plt.axvline(
        sentiment_data.mean(),
        linestyle="--",
        linewidth=2,
        label=(
            f"Mean = "
            f"{sentiment_data.mean():.2f}"
        ),
    )

    plt.xlabel(
        "Sentiment Score"
    )

    plt.ylabel(
        "Number of Interviews"
    )

    plt.title(
        "Distribution of Sentiment Scores in Pre-Tournament Interviews"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_sentiment_vs_performance(df):
    """
    Run point-biserial correlation and compare
    sentiment by performance group.
    """

    analysis_df = df[
        [
            SENTIMENT_COLUMN,
            PERFORMANCE_COLUMN,
        ]
    ].dropna()

    if analysis_df.empty:
        print(
            "Skipping performance analysis: "
            "no valid observations found."
        )
        return

    r, p_value = pointbiserialr(
        analysis_df[
            PERFORMANCE_COLUMN
        ],
        analysis_df[
            SENTIMENT_COLUMN
        ],
    )

    print(
        "\nNumber of interviews:",
        len(analysis_df),
    )

    print(
        f"Point-biserial correlation (r): "
        f"{r:.3f}"
    )

    print(
        f"P-value: "
        f"{p_value:.6f}"
    )

    mean_sentiment = (
        analysis_df
        .groupby(
            PERFORMANCE_COLUMN
        )[SENTIMENT_COLUMN]
        .mean()
    )

    print(
        "\nMean sentiment:"
    )

    print(
        mean_sentiment
    )

    below = analysis_df[
        analysis_df[
            PERFORMANCE_COLUMN
        ] == 0
    ][SENTIMENT_COLUMN]

    at_least = analysis_df[
        analysis_df[
            PERFORMANCE_COLUMN
        ] == 1
    ][SENTIMENT_COLUMN]

    plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        [
            below,
            at_least,
        ],
        tick_labels=[
            "Below Previous",
            "Same or Better then Previous",
        ],
    )

    plt.ylabel(
        "Interview Sentiment Score"
    )

    plt.xlabel(
        "Current Performance VS Previous Performance"
    )

    plt.title(
        "Pre-Tournament Sentiment by Performance Relative to Previous Tournaments"
    )

    plt.tight_layout()
    plt.show()


def run_sentiment_distribution():
    """Run sentiment distribution using the precomputed dataset."""

    df = load_data(
        SENTIMENT_DATASET
    )

    plot_sentiment_distribution(
        df
    )


def run_sentiment_performance():
    """Run sentiment-performance analysis using the precomputed dataset."""

    df = load_data(
        SENTIMENT_DATASET
    )

    analyze_sentiment_vs_performance(
        df
    )


# ============================================================
# Run all analyses
# ============================================================

def run_all_analysis():
    """
    Run all analyses using the provided precomputed sentiment scores.

    The full sentiment model and sanity check are not run.
    """

    original_df = load_data(
        ORIGINAL_DATASET
    )

    sentiment_df = load_data(
        SENTIMENT_DATASET
    )

    # General word analysis
    run_general_word_analysis(
        original_df
    )

    # Ranked-player word analysis
    run_ranked_word_analysis(
        original_df
    )

    # Sentiment distribution
    plot_sentiment_distribution(
        sentiment_df
    )

    # Sentiment vs. performance
    analyze_sentiment_vs_performance(
        sentiment_df
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tennis interview sentiment "
            "and word analysis"
        )
    )

    parser.add_argument(
        "--analysis",
        required=True,
        choices=[
            "words",
            "ranked-words",
            "sanity-check",
            "sentiment-model",
            "sentiment-distribution",
            "sentiment-performance",
            "all",
        ],
        help="Choose which analysis to run",
    )

    args = parser.parse_args()

    if args.analysis == "words":
        df = load_data(
            ORIGINAL_DATASET
        )

        run_general_word_analysis(
            df
        )

    elif args.analysis == "ranked-words":
        df = load_data(
            ORIGINAL_DATASET
        )

        run_ranked_word_analysis(
            df
        )

    elif args.analysis == "sanity-check":
        run_sanity_check()

    elif args.analysis == "sentiment-model":
        run_model_on_interviews()

    elif args.analysis == "sentiment-distribution":
        run_sentiment_distribution()

    elif args.analysis == "sentiment-performance":
        run_sentiment_performance()

    elif args.analysis == "all":
        run_all_analysis()


if __name__ == "__main__":
    main()