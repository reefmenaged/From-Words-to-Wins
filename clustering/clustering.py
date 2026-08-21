# ============================================================
# 1. Imports
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# 2. File paths and output directory
# ============================================================

PT_PATH = (
    r"../data/ckpt/test_interview_embeddings_after_attention.pt.zip"
)

CSV_PATH = (
    r"../data/processed/player_tournament_interview_dataset.csv"
)

RESULTS_DIR = "clustering_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 3. Load h_interview embeddings
# ============================================================
# The PyTorch file contains:
# - source_rows: original row indices of the 210 test interviews
# - embeddings: h_interview vector for each interview

data = torch.load(PT_PATH, map_location="cpu")

print("Available keys:", data.keys())
print("source_rows shape:", data["source_rows"].shape)
print("embeddings shape:", data["embeddings"].shape)


# ============================================================
# 4. Convert tensors to NumPy arrays
# ============================================================

source_rows = data["source_rows"].cpu().numpy()
X = data["embeddings"].cpu().numpy()

print("X shape:", X.shape)


# ============================================================
# 5. Sanity checks
# ============================================================
# Verify that the embeddings contain no invalid values.

print("NaN values:", np.isnan(X).sum())
print("Inf values:", np.isinf(X).sum())


# ============================================================
# 6. Normalize interview embeddings
# ============================================================
# Normalize each h_interview vector before computing cosine distance.

X_norm = normalize(X)


# ============================================================
# 7. Build the hierarchical clustering tree
# ============================================================
# Agglomerative hierarchical clustering:
# - cosine distance between interview embeddings
# - average linkage between clusters

Z = linkage(
    X_norm,
    method="average",
    metric="cosine"
)

print("Linkage matrix shape:", Z.shape)


# ============================================================
# 8. Experiment 1 - Dendrogram visualizations
# ============================================================
# Goal:
# Visualize the hierarchical structure of the interview embeddings
# both at a global level and at a zoomed-in level.

# ------------------------------------------------------------
# 8.1 Full dendrogram
# ------------------------------------------------------------

plt.figure(figsize=(14, 7))

dendrogram(
    Z,
    no_labels=True
)

plt.axhline(
    y=0.01,
    linestyle="--",
    linewidth=1,
    label="Coarse cut: t = 0.01"
)

plt.axhline(
    y=0.003,
    linestyle="--",
    linewidth=1,
    label="Fine cut: t = 0.003"
)

plt.title(
    "Hierarchical Clustering of Interview Representations (Full View)",
    fontsize=15
)
plt.xlabel("Interviews", fontsize=12)
plt.ylabel("Cosine Distance", fontsize=12)

plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "dendrogram_full.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 8.2 Zoomed-in dendrogram
# ------------------------------------------------------------
# Zoom into the lower part of the tree, where the meaningful
# fine-grained cluster separations occur.

plt.figure(figsize=(14, 7))

dendrogram(
    Z,
    no_labels=True
)

plt.axhline(
    y=0.01,
    linestyle="--",
    linewidth=2,
    label="Coarse cut: t = 0.01"
)

plt.axhline(
    y=0.003,
    linestyle="--",
    linewidth=2,
    label="Fine cut: t = 0.003"
)

plt.ylim(0, 0.02)

plt.title(
    "Hierarchical Clustering of Interview Representations (Zoomed View)",
    fontsize=15
)
plt.xlabel("Interviews", fontsize=12)
plt.ylabel("Cosine Distance", fontsize=12)

plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "dendrogram_zoomed.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# ============================================================
# 9. Experiment 2 - Examine different hierarchy levels
# ============================================================
# Goal:
# Cut the same dendrogram at several distance thresholds
# and compare the number and sizes of the resulting clusters.

thresholds = [0.10, 0.04, 0.02, 0.01]

summary_rows = []

for t in thresholds:

    labels = fcluster(
        Z,
        t=t,
        criterion="distance"
    )

    counts = pd.Series(labels).value_counts().sort_index()

    summary_rows.append({
        "distance_threshold": t,
        "num_clusters": len(counts),
        "largest_cluster": counts.max(),
        "smallest_cluster": counts.min()
    })

    print(f"\nThreshold = {t}")
    print("Number of clusters:", len(counts))
    print("Cluster sizes:")
    print(counts)


# Save threshold comparison

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    os.path.join(RESULTS_DIR, "threshold_summary.csv"),
    index=False
)

print("\nThreshold summary:")
print(summary_df)


# ============================================================
# 10. Save cluster assignments at selected hierarchy levels
# ============================================================
# Based on the dendrogram and Experiment 2, keep three
# representative levels of granularity.

selected_thresholds = {
    "level_1": 0.10,
    "level_2": 0.04,
    "level_3": 0.01
}

cluster_assignments = pd.DataFrame({
    "test_position": np.arange(len(source_rows)),
    "source_row": source_rows
})

for level_name, threshold in selected_thresholds.items():

    cluster_assignments[level_name] = fcluster(
        Z,
        t=threshold,
        criterion="distance"
    )

cluster_assignments.to_csv(
    os.path.join(RESULTS_DIR, "cluster_assignments.csv"),
    index=False
)

print("\nCluster assignments sample:")
print(cluster_assignments.head())


# ============================================================
# 11. Connect clusters to the original interview dataset
# ============================================================
# source_row connects each h_interview embedding back to the
# corresponding interview in the original dataset.

full_df = pd.read_csv(CSV_PATH)

test_df = (
    full_df
    .iloc[source_rows]
    .copy()
    .reset_index(drop=True)
)

test_df["source_row"] = source_rows

test_df["cluster_level_1"] = cluster_assignments["level_1"].to_numpy()
test_df["cluster_level_2"] = cluster_assignments["level_2"].to_numpy()
test_df["cluster_level_3"] = cluster_assignments["level_3"].to_numpy()

print("\nTest dataframe shape:", test_df.shape)


# ============================================================
# 12. Inspect the final clustering level
# ============================================================
# At threshold 0.01 we obtained four clusters.
# First, examine their sizes.

cluster_sizes = (
    test_df["cluster_level_3"]
    .value_counts()
    .sort_index()
)

print("\nCluster sizes at threshold 0.01:")
print(cluster_sizes)


# Save all interview-cluster assignments instead of printing
# all 210 interviews to the console.

columns_to_save = [
    "source_row",
    "player_name",
    "tourney_name",
    "tourney_year",
    "player_rank",
    "tournament_finish_score",
    "current_finish_at_least_recent_average",
    "num_questions",
    "avg_answer_length_words",
    "total_answer_words",
    "cluster_level_3"
]

test_df[columns_to_save].to_csv(
    os.path.join(RESULTS_DIR, "interviews_with_clusters.csv"),
    index=False
)


# ============================================================
# 13. Inspect small and unusual clusters
# ============================================================
# Clusters with very few interviews may represent unusual
# interview representations and should be inspected separately.

small_cluster_ids = cluster_sizes[
    cluster_sizes <= 25
].index

small_cluster_df = test_df[
    test_df["cluster_level_3"].isin(small_cluster_ids)
]

print("\nInterviews in smaller clusters:")
print(
    small_cluster_df[columns_to_save]
    .sort_values("cluster_level_3")
    .to_string(index=False)
)


# ============================================================
# 14. Compare numerical characteristics between clusters
# ============================================================
# Compare basic interview, player and performance characteristics
# to understand what may distinguish the clusters.

cluster_summary = (
    test_df
    .groupby("cluster_level_3")
    .agg(
        num_interviews=("source_row", "count"),
        success_rate=(
            "current_finish_at_least_recent_average",
            "mean"
        ),
        avg_finish_score=(
            "tournament_finish_score",
            "mean"
        ),
        avg_player_rank=(
            "player_rank",
            "mean"
        ),
        avg_num_questions=(
            "num_questions",
            "mean"
        ),
        avg_answer_length=(
            "avg_answer_length_words",
            "mean"
        ),
        avg_total_words=(
            "total_answer_words",
            "mean"
        )
    )
)

print("\nCluster summary:")
print(cluster_summary.to_string())

cluster_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "cluster_level3_summary.csv"
    )
)


# ============================================================
# 15. Experiment 3 - Interview length across clusters
# ============================================================
# Motivation:
# Initial inspection suggested that the clusters may differ
# substantially in interview length.
#
# Goal:
# Compare the average number of answer words across clusters.

cluster_avg_words = (
    test_df
    .groupby("cluster_level_3")["total_answer_words"]
    .mean()
)

cluster_counts = (
    test_df
    .groupby("cluster_level_3")
    .size()
)

# Include cluster size in the x-axis labels.
x_labels = [
    f"Cluster {cluster}\n(n={cluster_counts[cluster]})"
    for cluster in cluster_avg_words.index
]

plt.figure(figsize=(8, 6))

plt.bar(
    range(len(cluster_avg_words)),
    cluster_avg_words.values
)

plt.xticks(
    range(len(cluster_avg_words)),
    x_labels
)

plt.title("Average Interview Length by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Total Answer Words")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "average_interview_length_by_cluster.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 16. Experiment 4 - Inspect potential outliers
# ============================================================
# Goal:
# Examine the interviews that formed very small clusters and determine
# whether they represent valid but unusual interviews or possible
# data-quality / parsing problems.


# ------------------------------------------------------------
# 16.1 Show the shortest interviews in the test set
# ------------------------------------------------------------

shortest_interviews = (
    test_df[
        [
            "source_row",
            "player_name",
            "tourney_name",
            "num_questions",
            "avg_answer_length_words",
            "total_answer_words",
            "cluster_level_3"
        ]
    ]
    .sort_values("total_answer_words")
    .head(15)
)

print("\n15 shortest interviews:")
print(shortest_interviews.to_string(index=False))

# ============================================================
# 17. Inspect the shortest interview
# ============================================================
# The shortest interview contains only 26 answer words.
# We inspect its original text to check whether it is a valid
# interview or whether there may be a data collection/parsing issue.

shortest_interview = test_df[
    test_df["source_row"] == 751
].iloc[0]

print("\nShortest interview:")
print("Player:", shortest_interview["player_name"])
print("Tournament:", shortest_interview["tourney_name"])
print("Number of questions:", shortest_interview["num_questions"])
print("Total answer words:", shortest_interview["total_answer_words"])

print("\nRaw interview:")
print(shortest_interview["first_pre_match_interview_qa_json"])

# ============================================================
# 18. Find the interviews closest to the shortest interview
# ============================================================
# The shortest interview is valid but forms a cluster by itself.
# We now check which interview embeddings are most similar to it
# according to cosine distance.

outlier_source_row = 751

# Find its position inside the test set
outlier_position = np.where(source_rows == outlier_source_row)[0][0]

# Get its normalized h_interview vector
outlier_vector = X_norm[outlier_position]

# Calculate cosine distance from this interview to all interviews
distances = 1 - np.dot(X_norm, outlier_vector)

# Ignore distance to itself
distances[outlier_position] = np.inf

# Find the 5 closest interviews
nearest_positions = np.argsort(distances)[:5]

nearest_rows = []

for position in nearest_positions:

    nearest_rows.append({
        "source_row": source_rows[position],
        "player_name": test_df.iloc[position]["player_name"],
        "total_answer_words": test_df.iloc[position]["total_answer_words"],
        "cluster": test_df.iloc[position]["cluster_level_3"],
        "cosine_distance": distances[position]
    })

nearest_df = pd.DataFrame(nearest_rows)

print("\nClosest interviews to source_row 751:")
print(nearest_df.to_string(index=False))

# ============================================================
# 19. Compare the outlier with its closest interview
# ============================================================

rows_to_compare = [751, 574]

for row_id in rows_to_compare:

    interview = test_df[
        test_df["source_row"] == row_id
    ].iloc[0]

    print("\n" + "=" * 80)
    print("Source row:", row_id)
    print("Player:", interview["player_name"])
    print("Cluster:", interview["cluster_level_3"])
    print("Total words:", interview["total_answer_words"])

    print("\nRaw interview:")
    print(interview["first_pre_match_interview_qa_json"])

# ============================================================
# 20. Experiment 5 - Compare interview structure
# ============================================================
# Goal:
# Understand why Cluster 2 contains much longer interviews
# than Cluster 3.
# We compare the average number of questions and the average
# answer length between the two main clusters.

main_clusters = test_df[
    test_df["cluster_level_3"].isin([2, 3])
]

structure_summary = (
    main_clusters
    .groupby("cluster_level_3")
    .agg(
        num_interviews=("source_row", "count"),
        avg_num_questions=("num_questions", "mean"),
        avg_answer_length_words=("avg_answer_length_words", "mean"),
        avg_total_words=("total_answer_words", "mean")
    )
)

print("\nInterview structure - Clusters 2 and 3:")
print(structure_summary.to_string())


# ============================================================
# 21. Experiment 6 - Find a representative interview per cluster
# ============================================================
# Goal:
# Find an actual interview that best represents each main cluster.
#
# The representative interview (clustroid) is the interview whose
# embedding has the smallest average cosine distance to all other
# interviews in the same cluster.

for cluster_id in [2, 3]:

    # Positions of interviews belonging to this cluster
    cluster_positions = np.where(
        test_df["cluster_level_3"].to_numpy() == cluster_id
    )[0]

    # Normalized embeddings of the cluster
    cluster_vectors = X_norm[cluster_positions]

    # Cosine distance between every pair of interviews in the cluster
    distance_matrix = 1 - np.dot(
        cluster_vectors,
        cluster_vectors.T
    )

    # Average distance of each interview to all other interviews
    avg_distances = distance_matrix.mean(axis=1)

    # Interview with the smallest average distance = clustroid
    representative_local_position = np.argmin(avg_distances)

    representative_position = cluster_positions[
        representative_local_position
    ]

    representative = test_df.iloc[representative_position]

    print("\n" + "=" * 80)
    print(f"Representative interview - Cluster {cluster_id}")
    print("=" * 80)

    print("Source row:", representative["source_row"])
    print("Player:", representative["player_name"])
    print("Tournament:", representative["tourney_name"])
    print("Total words:", representative["total_answer_words"])
    print("Average distance inside cluster:",
          avg_distances[representative_local_position])

    print("\nInterview:")
    print(representative["first_pre_match_interview_qa_json"])


# ============================================================
# 22. Experiment 7 - Find the 5 most representative interviews
# ============================================================
# Goal:
# Examine several representative interviews from each main cluster
# to look for recurring semantic/content patterns.

for cluster_id in [2, 3]:

    cluster_positions = np.where(
        test_df["cluster_level_3"].to_numpy() == cluster_id
    )[0]

    cluster_vectors = X_norm[cluster_positions]

    # Pairwise cosine distances inside the cluster
    distance_matrix = 1 - np.dot(
        cluster_vectors,
        cluster_vectors.T
    )

    # Average distance of each interview to the rest of the cluster
    avg_distances = distance_matrix.mean(axis=1)

    # Five interviews with the smallest average distance
    five_best_local = np.argsort(avg_distances)[:5]

    print("\n" + "=" * 80)
    print(f"5 most representative interviews - Cluster {cluster_id}")
    print("=" * 80)

    for local_position in five_best_local:

        global_position = cluster_positions[local_position]
        interview = test_df.iloc[global_position]

        print(
            "Source row:", interview["source_row"],
            "| Player:", interview["player_name"],
            "| Tournament:", interview["tourney_name"],
            "| Avg distance:", round(avg_distances[local_position], 6)
        )

# ============================================================
# 23. Experiment 8 - Gender distribution across main clusters
# ============================================================

gender_counts = pd.crosstab(
    test_df["cluster_level_3"],
    test_df["is_male"]
)

# Compare only the two main clusters
gender_counts = gender_counts.loc[[2, 3]]

# Rename columns for readability
gender_counts = gender_counts.rename(
    columns={
        0: "Female",
        1: "Male"
    }
)

print("\nGender distribution - Clusters 2 and 3:")
print(gender_counts)

# Convert counts to percentages
gender_percentages = (
    gender_counts
    .div(gender_counts.sum(axis=1), axis=0)
    * 100
)

print("\nGender percentages - Clusters 2 and 3:")
print(gender_percentages.round(1))


# ============================================================
# 25. Experiment 9 - Cluster distribution for repeated players
# ============================================================
# Goal:
# Check whether interviews from the same player consistently
# appear in the same cluster.

main_clusters = test_df[
    test_df["cluster_level_3"].isin([2, 3])
]

player_cluster_counts = pd.crosstab(
    main_clusters["player_name"],
    main_clusters["cluster_level_3"]
)

# Keep only players with at least 3 interviews
player_cluster_counts["total"] = player_cluster_counts.sum(axis=1)

repeated_players = player_cluster_counts[
    player_cluster_counts["total"] >= 3
].copy()

# Percentage of each player's interviews in each cluster
repeated_players["cluster_2_percent"] = (
    repeated_players[2] / repeated_players["total"] * 100
)

repeated_players["cluster_3_percent"] = (
    repeated_players[3] / repeated_players["total"] * 100
)

print("\nCluster distribution for repeated players:")
print(
    repeated_players[
        [2, 3, "total", "cluster_2_percent", "cluster_3_percent"]
    ]
    .sort_values("total", ascending=False)
    .to_string()
)


# ============================================================
# 26. Experiment 10 - Prepare interview text for content analysis
# ============================================================
# Goal:
# Extract only the players' answers from each interview,
# so that we can later compare the textual content of the clusters.

import json


def extract_answers(qa_json):

    data = json.loads(qa_json)

    answers = []

    for key, value in data.items():

        if key.startswith("answer_"):
            answers.append(value)

    return " ".join(answers)


test_df["answers_text"] = (
    test_df["first_pre_match_interview_qa_json"]
    .apply(extract_answers)
)


# Show a few examples to verify that the extraction worked correctly
print("\nExtracted answers:")
print(
    test_df[
        ["player_name", "cluster_level_3", "answers_text"]
    ]
    .head(3)
    .to_string(index=False)
)

# ============================================================
# 27. Verify extracted text in both main clusters
# ============================================================

for cluster_id in [2, 3]:

    example = test_df[
        test_df["cluster_level_3"] == cluster_id
    ].iloc[0]

    print("\n" + "=" * 80)
    print(f"Example from Cluster {cluster_id}")
    print("Player:", example["player_name"])
    print("=" * 80)

    # Print only the first 500 characters
    print(example["answers_text"][:500])


# ============================================================
# 28. Experiment 11 - Compare textual patterns between clusters
# ============================================================
# Goal:
# Find words and short phrases that are more characteristic
# of Cluster 2 or Cluster 3.

main_clusters = test_df[
    test_df["cluster_level_3"].isin([2, 3])
].copy()

# Convert the interview answers into TF-IDF features.
# English stop words such as "the", "and", "is" are removed.
# We consider both single words and two-word phrases.
custom_stop_words = {
    "like",
    "know",
    "think",
    "obviously",
    "just",
    "things",
    "kind",
    "don",
    "maybe",
    "mean",
    "yeah",
    "really"
}

stop_words = list(
    ENGLISH_STOP_WORDS.union(custom_stop_words)
)

vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=(2, 2),
    min_df=2
)

tfidf_matrix = vectorizer.fit_transform(
    main_clusters["answers_text"]
)

terms = np.array(
    vectorizer.get_feature_names_out()
)

# Positions of interviews from each cluster
cluster_2_mask = (
    main_clusters["cluster_level_3"].to_numpy() == 2
)

cluster_3_mask = (
    main_clusters["cluster_level_3"].to_numpy() == 3
)

# Average TF-IDF value of every word/phrase in each cluster
cluster_2_mean = np.asarray(
    tfidf_matrix[cluster_2_mask].mean(axis=0)
).ravel()

cluster_3_mean = np.asarray(
    tfidf_matrix[cluster_3_mask].mean(axis=0)
).ravel()

# Difference between the clusters
difference = cluster_2_mean - cluster_3_mean

# Terms most characteristic of Cluster 2
top_cluster_2 = np.argsort(difference)[-15:][::-1]

# Terms most characteristic of Cluster 3
top_cluster_3 = np.argsort(difference)[:15]


print("\nMost characteristic terms - Cluster 2:")
for i in top_cluster_2:
    print(terms[i], round(difference[i], 4))


print("\nMost characteristic terms - Cluster 3:")
for i in top_cluster_3:
    print(terms[i], round(-difference[i], 4))


# ============================================================
# 29. Experiment 12 - Readiness/preparation language
# ============================================================
# Goal:
# Check whether readiness and preparation language appears
# more frequently in Cluster 3 than in Cluster 2.

readiness_terms = [
    "ready",
    "preparation",
    "prepare",
    "prepared",
    "looking forward",
    "pre-season",
    "pre season",
    "excited"
]


def contains_readiness_language(text):

    text = text.lower()

    return any(
        term in text
        for term in readiness_terms
    )


main_clusters["readiness_language"] = (
    main_clusters["answers_text"]
    .apply(contains_readiness_language)
)


readiness_summary = (
    main_clusters
    .groupby("cluster_level_3")["readiness_language"]
    .agg(["count", "sum", "mean"])
)

readiness_summary["percentage"] = (
    readiness_summary["mean"] * 100
)

print("\nReadiness/preparation language:")
print(readiness_summary.to_string())

# ============================================================
# 30. Experiment 13 - Positive / enthusiastic language
# ============================================================
# Goal:
# Check whether Cluster 3 contains more positive or enthusiastic
# language than Cluster 2.

positive_terms = [
    "happy",
    "excited",
    "looking forward",
    "fun",
    "enjoy",
    "love",
    "positive",
    "confident",
    "great"
]


def contains_positive_language(text):

    text = text.lower()

    return any(
        term in text
        for term in positive_terms
    )


main_clusters["positive_language"] = (
    main_clusters["answers_text"]
    .apply(contains_positive_language)
)


positive_summary = (
    main_clusters
    .groupby("cluster_level_3")["positive_language"]
    .agg(["count", "sum", "mean"])
)

positive_summary["percentage"] = (
    positive_summary["mean"] * 100
)

print("\nPositive / enthusiastic language:")
print(positive_summary.to_string())

# ============================================================
# 31. Experiment 14 - Explore finer clustering levels
# ============================================================
# Goal:
# Examine finer levels of the hierarchical tree to see whether
# the large cluster splits into smaller and more informative groups.

fine_thresholds = [0.008, 0.006, 0.004, 0.003, 0.002]

for t in fine_thresholds:

    labels = fcluster(
        Z,
        t=t,
        criterion="distance"
    )

    counts = pd.Series(labels).value_counts().sort_values(ascending=False)

    print("\n" + "=" * 60)
    print(f"Threshold = {t}")
    print("Number of clusters:", len(counts))
    print("Cluster sizes:")
    print(counts.to_string())

# ============================================================
# 32. Experiment 15 - Select finer clustering level
# ============================================================
# Threshold 0.003 produces two large clusters and several
# smaller groups, allowing a more detailed content analysis.

fine_threshold = 0.003

fine_labels = fcluster(
    Z,
    t=fine_threshold,
    criterion="distance"
)

test_df["cluster_fine"] = fine_labels

fine_cluster_sizes = (
    test_df["cluster_fine"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nCluster sizes at threshold 0.003:")
print(fine_cluster_sizes)

# ============================================================
# 33. Experiment 16 - Representative interviews in fine clusters
# ============================================================
# Goal:
# Find the 5 most representative interviews in the two largest
# clusters at threshold 0.003.

for cluster_id in [5, 4]:

    # Positions of interviews belonging to the cluster
    cluster_positions = np.where(
        test_df["cluster_fine"].to_numpy() == cluster_id
    )[0]

    # Their normalized h_interview vectors
    cluster_vectors = X_norm[cluster_positions]

    # Pairwise cosine distances
    distance_matrix = 1 - np.dot(
        cluster_vectors,
        cluster_vectors.T
    )

    # Average distance of each interview to the rest of its cluster
    avg_distances = distance_matrix.mean(axis=1)

    # Five most central / representative interviews
    five_best_local = np.argsort(avg_distances)[:5]

    print("\n" + "=" * 80)
    print(f"5 most representative interviews - Fine Cluster {cluster_id}")
    print("=" * 80)

    for local_position in five_best_local:

        global_position = cluster_positions[local_position]
        interview = test_df.iloc[global_position]

        print(
            "Source row:", interview["source_row"],
            "| Player:", interview["player_name"],
            "| Tournament:", interview["tourney_name"],
            "| Words:", interview["total_answer_words"],
            "| Avg distance:", round(avg_distances[local_position], 6)
        )


# ============================================================
# 34. Experiment 17 - Compare textual patterns in fine clusters
# ============================================================
# Goal:
# Find two-word phrases that are more characteristic of
# Fine Cluster 5 compared with Fine Cluster 4, and vice versa.

fine_main_clusters = test_df[
    test_df["cluster_fine"].isin([4, 5])
].copy()


fine_vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=(2, 2),
    min_df=2
)

fine_tfidf = fine_vectorizer.fit_transform(
    fine_main_clusters["answers_text"]
)

fine_terms = np.array(
    fine_vectorizer.get_feature_names_out()
)


# Identify interviews from each cluster
cluster_5_mask = (
    fine_main_clusters["cluster_fine"].to_numpy() == 5
)

cluster_4_mask = (
    fine_main_clusters["cluster_fine"].to_numpy() == 4
)


# Average TF-IDF score of every phrase in each cluster
cluster_5_mean = np.asarray(
    fine_tfidf[cluster_5_mask].mean(axis=0)
).ravel()

cluster_4_mean = np.asarray(
    fine_tfidf[cluster_4_mask].mean(axis=0)
).ravel()


# Difference between the clusters
difference = cluster_5_mean - cluster_4_mean


# 15 phrases most characteristic of Cluster 5
top_cluster_5 = np.argsort(difference)[-15:][::-1]

# 15 phrases most characteristic of Cluster 4
top_cluster_4 = np.argsort(difference)[:15]


print("\nMost characteristic phrases - Fine Cluster 5:")
for i in top_cluster_5:
    print(fine_terms[i], round(difference[i], 4))


print("\nMost characteristic phrases - Fine Cluster 4:")
for i in top_cluster_4:
    print(fine_terms[i], round(-difference[i], 4))


# ============================================================
# 35. Inspect representative interviews from Fine Clusters 4 and 5
# ============================================================
# Goal:
# Read short excerpts from representative interviews and check
# whether the textual differences suggested by TF-IDF are visible
# in the actual interviews.

for cluster_id in [5, 4]:

    cluster_positions = np.where(
        test_df["cluster_fine"].to_numpy() == cluster_id
    )[0]

    cluster_vectors = X_norm[cluster_positions]

    distance_matrix = 1 - np.dot(
        cluster_vectors,
        cluster_vectors.T
    )

    avg_distances = distance_matrix.mean(axis=1)

    five_best_local = np.argsort(avg_distances)[:5]

    print("\n" + "=" * 80)
    print(f"Representative excerpts - Fine Cluster {cluster_id}")
    print("=" * 80)

    for local_position in five_best_local:

        global_position = cluster_positions[local_position]
        interview = test_df.iloc[global_position]

        print("\nPlayer:", interview["player_name"])
        print("Tournament:", interview["tourney_name"])

        # Print only the first 700 characters of the player's answers
        print(interview["answers_text"][:700])

        print("-" * 80)


# ============================================================
# 36. Experiment 18 - Phrase occurrence across fine clusters
# ============================================================
# Goal:
# Find two-word phrases that appear in a larger percentage
# of interviews in one cluster compared with the other.

phrase_vectorizer = CountVectorizer(
    stop_words=stop_words,
    ngram_range=(2, 2),
    min_df=3,
    binary=True
)

phrase_matrix = phrase_vectorizer.fit_transform(
    fine_main_clusters["answers_text"]
)

phrases = np.array(
    phrase_vectorizer.get_feature_names_out()
)

cluster_5_mask = (
    fine_main_clusters["cluster_fine"].to_numpy() == 5
)

cluster_4_mask = (
    fine_main_clusters["cluster_fine"].to_numpy() == 4
)

# Because the matrix is binary, the mean tells us
# the proportion of interviews containing each phrase.
cluster_5_percent = np.asarray(
    phrase_matrix[cluster_5_mask].mean(axis=0)
).ravel() * 100

cluster_4_percent = np.asarray(
    phrase_matrix[cluster_4_mask].mean(axis=0)
).ravel() * 100

difference = cluster_5_percent - cluster_4_percent

top_5 = np.argsort(difference)[-15:][::-1]
top_4 = np.argsort(difference)[:15]


print("\nPhrases more common in Fine Cluster 5:")
for i in top_5:
    print(
        phrases[i],
        "| Cluster 5:", round(cluster_5_percent[i], 1), "%",
        "| Cluster 4:", round(cluster_4_percent[i], 1), "%"
    )


print("\nPhrases more common in Fine Cluster 4:")
for i in top_4:
    print(
        phrases[i],
        "| Cluster 4:", round(cluster_4_percent[i], 1), "%",
        "| Cluster 5:", round(cluster_5_percent[i], 1), "%"
    )

# ============================================================
# 37. Visualize characteristic phrases in Clusters 4 and 5
# ============================================================
# Goal:
# Compare the percentage of interviews containing selected
# characteristic phrases in the two main clusters at threshold 0.003.

selected_phrases = [
    "grand slams",
    "couple years",
    "tennis player",
    "ve learned",
    "feeling good",
    "going good",
    "doing great",
    "looking forward"
]

phrase_results = []

for phrase in selected_phrases:

    i = np.where(phrases == phrase)[0][0]

    phrase_results.append({
        "phrase": phrase,
        "Cluster 5 (n=103)": cluster_5_percent[i],
        "Cluster 4 (n=78)": cluster_4_percent[i]
    })


# Create dataframe
phrase_df = pd.DataFrame(
    phrase_results
).set_index("phrase")


# Improve display labels
phrase_df = phrase_df.rename(
    index={
        "ve learned": "have learned"
    }
)


# Print exact percentages
print("\nSelected phrase percentages:")
print(phrase_df)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

ax = phrase_df.plot(
    kind="barh",
    figsize=(10, 6)
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3,
        fontsize=9
    )

plt.title(
    "Characteristic Phrase Occurrence by Cluster",
    fontsize=15
)

plt.xlabel(
    "Interviews Containing the Phrase (%)",
    fontsize=12
)

plt.ylabel("")

plt.legend(
    title="Cluster",
    fontsize=10,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)

plt.xlim(0, 35)

plt.tight_layout()


# Save graph
plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "characteristic_phrases_final.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 38. Experiment 19 - Performance comparison between Fine Clusters
# ============================================================
# Goal:
# Check whether Fine Clusters 4 and 5 also differ in tournament
# performance, in addition to their textual differences.

performance_summary = (
    test_df[
        test_df["cluster_fine"].isin([4, 5])
    ]
    .groupby("cluster_fine")
    .agg(
        num_interviews=("source_row", "count"),

        success_rate=(
            "current_finish_at_least_recent_average",
            "mean"
        ),

        avg_finish_score=(
            "tournament_finish_score",
            "mean"
        ),

        median_finish_score=(
            "tournament_finish_score",
            "median"
        )
    )
)

# Convert success rate to percentage
performance_summary["success_rate_percent"] = (
    performance_summary["success_rate"] * 100
)

print("\nPerformance comparison - Fine Clusters 4 and 5:")
print(performance_summary.to_string())


# ============================================================
# 39. Visualize relative tournament success in Clusters 4 and 5
# ============================================================
# Success = current tournament finish is at least as good as
# the player's average finish in the previous 3 tournaments.

success_rates = (
    test_df[
        test_df["cluster_fine"].isin([4, 5])
    ]
    .groupby("cluster_fine")[
        "current_finish_at_least_recent_average"
    ]
    .mean()
    * 100
)

success_rates.index = [
    "Cluster 4 (n=78)",
    "Cluster 5 (n=103)"
]


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))

bars = ax.barh(
    success_rates.index,
    success_rates.values,
    height=0.36
)

# Put Cluster 4 on top
ax.invert_yaxis()


# Main title
ax.set_title(
    "Relative Tournament Success Rate by Cluster",
    fontsize=15,
    pad=28
)

# Short explanation under the title
ax.text(
    0.5,
    1.04,
    "Success = current tournament finish ≥ average of previous 3 tournament finishes",
    transform=ax.transAxes,
    ha="center",
    fontsize=9
)


# Axis labels
ax.set_xlabel(
    "Interviews Followed by Relative Success (%)",
    fontsize=11
)

ax.set_ylabel("")


# Percentage axis
ax.set_xlim(0, 60)
ax.set_xticks(np.arange(0, 61, 10))


# Light grid lines
ax.xaxis.grid(
    True,
    linestyle="--",
    linewidth=0.8,
    alpha=0.18
)

ax.set_axisbelow(True)


# Exact values at the end of each bar
ax.bar_label(
    bars,
    labels=[f"{value:.1f}%" for value in success_rates.values],
    padding=8,
    fontsize=11,
    fontweight="bold"
)


# Improve spacing and readability
ax.tick_params(
    axis="both",
    labelsize=10
)

ax.margins(y=0.22)


# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.tight_layout()


# Save graph
plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "cluster_relative_success_rate_final.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 40. 2D visualization of all fine clusters (threshold = 0.003)
# ============================================================

from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import os

# Reduce the normalized embeddings to 2D
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_norm)

# Create plotting dataframe
plot_df = pd.DataFrame({
    "dim1": X_2d[:, 0],
    "dim2": X_2d[:, 1],
    "cluster_fine": test_df["cluster_fine"].values,
    "label": test_df["current_finish_at_least_recent_average"].astype(int).values
})

# Get cluster sizes
cluster_sizes = (
    plot_df["cluster_fine"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nCluster sizes at threshold 0.003:")
print(cluster_sizes)

# Create figure
plt.figure(figsize=(11, 8))

# Plot each cluster in a different color
clusters_sorted = sorted(plot_df["cluster_fine"].unique())

for cluster_id in clusters_sorted:
    subset = plot_df[plot_df["cluster_fine"] == cluster_id]

    plt.scatter(
        subset["dim1"],
        subset["dim2"],
        s=55,
        alpha=0.75,
        label=f"Cluster {cluster_id} (n={len(subset)})"
    )

plt.title("2D Projection of All Fine Clusters (threshold = 0.003)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(alpha=0.3)
plt.legend(
    title="Fine Clusters",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "fine_clusters_2d_all_clusters.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()