"""
This script performs association rule mining on a market basket dataset.

It uses the Apriori algorithm to find frequent itemsets, generates association
rules, prints them, and visualizes the rules as a directed graph.

Author: vnniciusg
"""

import os
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from kagglehub import dataset_download
from loguru import logger
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")

_MIN_SUPPORT: float = 2e-2


def load_df(handle: str = "rukenmissonnier/real-market-data") -> pd.DataFrame:
    """
    Downloads a dataset from Kaggle and loads it into a pandas DataFrame.

    Args:
        handle (str): The Kaggle dataset handle.

    Returns:
        pd.DataFrame: The loaded DataFrame, with all columns cast to bool.

    Raises:
        FileNotFoundError: If no CSV file is found in the downloaded dataset.
    """
    path = dataset_download(handle)
    files = os.listdir(path)

    for file in files:
        if file.endswith(".csv"):
            return pd.read_csv(os.path.join(path, file), sep=";").astype(bool)

    raise FileNotFoundError("No CSV file found in the downloaded dataset.")


def perform_association_rule_mining(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs association rule mining on the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with transaction data.

    Returns:
        pd.DataFrame: A DataFrame containing the association rules.
    """
    frequent_itemsets = apriori(df, min_support=_MIN_SUPPORT, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    return rules.sort_values(["confidence", "lift"], ascending=False)


def visualize_association_rules(
    rules: pd.DataFrame, output_filename: str = "association_rules.png", top_n: int = 5
):
    """
    Visualizes the association rules as a directed graph and saves it to a file.

    Args:
        rules (pd.DataFrame): DataFrame containing the association rules.
        output_filename (str): The name of the output image file.
        top_n (int): The number of top rules to vizualize
    """

    rules_to_plot = rules.head(top_n)
    G = nx.DiGraph()

    for _, rule in rules_to_plot.iterrows():
        G.add_edge(rule["antecedents"], rule["consequents"], weight=rule["lift"])

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=1.5)

    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color="lightgreen",
        node_size=3000,
        edge_color="gray",
    )
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2"),
    )

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}
    )

    plt.title("Top Association Rules Network", size=15)
    plt.savefig(output_filename)
    plt.close()

    logger.success(f"Graph visualization saved to {output_filename}")


def main():
    try:
        df = load_df()
        rules = perform_association_rule_mining(df=df)
        logger.info("Association Rules:")
        logger.info(rules.head())

        if rules.empty:
            logger.warning("No association rules found with the given parameters.")
            return

        visualize_association_rules(rules)

    except FileNotFoundError as e:
        logger.error(e)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
