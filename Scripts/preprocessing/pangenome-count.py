#!/usr/bin/env python3

import sys
from argparse import ArgumentParser

import pandas as pd

def format_series(series: pd.Series) -> pd.Series:
    """Removes double quotes and counts the number of genes per table cell."""
    return series.str.replace("\"", "").str.strip().str.count(" ")

def main():

    # Create parser
    parser = ArgumentParser(
        description=(
            "Print a gene count TSV table from a PPanGGOLiN matrix.csv file."
        ),
        epilog="Example usage: ./pangenome-count.py matrix.csv > output.tsv"
    )
    parser.add_argument("file", help="PPanGGOLiN matrix.csv file")
    parser.add_argument(
        "-a", "--antibiotic", required=True,
        choices=["ciprofloxacin", "meropenem"],
        help="type of antibiotic"
    )

    # Print help message if no arguments are given or parse them otherwise
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    file, antibiotic = map(str, vars(parser.parse_args()).values())

    # Load matrix
    print("Loading matrix...", file=sys.stderr, end="\r")

    matrix = pd.read_csv(file, index_col="Gene", dtype=str)

    # Preprocess matrix
    print("Preprocessing matrix...", file=sys.stderr, end="\r")

    # Edit matrix so it stores gene counts instead of listing individual genes
    matrix = (
        matrix[matrix.columns[13:]].apply(format_series).astype(float) + 1
    ).fillna(0).T

    # Simplify index column
    matrix.index = (
        matrix.index.str.split("_", n=2).str.get(2)
        .str.removesuffix(".1").str.removesuffix(".2")
    )
    matrix.index.name = "accession"

    # Load metadata and merge it into matrix
    print("Loading and merging metadata...", file=sys.stderr, end="\r")

    training = pd.read_csv(
        "https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/"
        "aca2b4bef8c642f7ccd5adb63e7f054feb67783a/DataSets/TrainingDataset.csv",
        index_col="accession",
        usecols=list(range(4)) + [6]
    )
    testing = pd.read_csv(
        "https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/"
        "d105fa8d90f8b3c397958d09d1fa615597883459/DataSets/TestingDataset.csv",
        index_col="accession",
        usecols=list(range(4))
    )
    testing = (
        testing[testing["antibiotic"] == antibiotic].drop("antibiotic", axis=1)
    )
    metadata = pd.concat([training, testing])
    metadata.index = (
        metadata.index.str.removesuffix(".1").str.removesuffix(".2")
    )

    matrix = pd.merge(metadata, matrix, left_index=True, right_index=True)

    # Print matrix
    print("Writing output matrix...", file=sys.stderr, end="\r")
    matrix.to_csv(sys.stdout, sep="\t")

if __name__ == "__main__":
    main()
