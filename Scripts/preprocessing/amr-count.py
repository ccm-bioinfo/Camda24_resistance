#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

def main():

    # Create parser
    parser = ArgumentParser(
        description=(
            "Print a gene count TSV table from RGI main tab delimited files."),
        epilog="Example usage: ./amr-count.py *.tsv > output.tsv"
    )
    parser.add_argument(
        "file", help="RGI main tab delimited file", nargs="+", type=Path)
    parser.add_argument(
        "-s", "--strict", help="do not include loose hits", action="store_true")

    # Print help message if no arguments are given or parse them otherwise
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    files, strict = vars(parser.parse_args()).values()
    count = 0

    # Function that processes a single RGI main tab delimited file
    def process_tsv(file: Path, strict: bool) -> pd.Series:
        nonlocal count
        table = pd.read_csv(file, sep="\t", usecols=["Cut_Off", "ARO"])
        if strict: table = table[table["Cut_Off"] != "Loose"]
        output = table["ARO"].value_counts()
        output.name = file.stem.removesuffix(".1").removesuffix(".2")
        count += 1
        print(f"{count / len(files) * 100:.2f}%", end="\r", file=sys.stderr)
        return output

    # Build output tab delimited table
    series = map(lambda file: process_tsv(file, strict), files)
    table = pd.concat(series, axis=1).fillna(0).T
    table.index.name = "accession"
    print("Writing...", end="\r", file=sys.stderr)

    # Read metadata table and merge it into main table
    training = pd.read_csv(
        "https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/"
        "aca2b4bef8c642f7ccd5adb63e7f054feb67783a/DataSets/TrainingDataset.csv",
        index_col="accession",
        usecols=list(range(5)) + [6]
    )
    testing = pd.read_csv(
        "https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/"
        "d105fa8d90f8b3c397958d09d1fa615597883459/DataSets/TestingDataset.csv",
        index_col="accession",
        usecols=list(range(4))
    )
    metadata = pd.concat([training, testing])
    metadata.index = (
        metadata.index.str.removesuffix(".1").str.removesuffix(".2")
    )
    table = pd.merge(metadata, table, left_index=True, right_index=True)

    # Print table
    table.to_csv(sys.stdout, sep="\t")

if __name__ == "__main__":
    main()
