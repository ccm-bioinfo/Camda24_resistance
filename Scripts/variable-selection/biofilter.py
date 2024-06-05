#!/usr/bin/env python3

"""Filters resistance count tables to only genes that confer resistance to
meropenem or ciprofloxacin."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

def parse_args() -> str:
    """Creates a parser for script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", help="path to DataSets directory")
    if len(sys.argv) < 2:
        parser.print_help()
        exit(1)
    return parser.parse_args().datasets

def main():
    # Read DataSets directory from parameter, and setup metadata and file dirs
    datasets = parse_args()
    metadata_file = f"{datasets}/ResistanceCategories.tsv"
    files = [
        Path(f"{datasets}/ResistanceGeneCountLoose.tsv.gz"),
        Path(f"{datasets}/ResistanceGeneCountStrict.tsv.gz"),
        Path(f"{datasets}/ResistanceSNPCountLoose.tsv.gz"),
        Path(f"{datasets}/ResistanceSNPCountStrict.tsv.gz")
    ]
    output = Path(f"{datasets}/SelectedVariables")

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Read metadata table and filter to gene AROs of interest
    metadata = pd.read_csv(
        metadata_file, sep="\t", index_col="ARO", usecols=["ARO", "Antibiotic"]
    )["Antibiotic"].str.split(", ").explode()
    aro = set(
        metadata[metadata.isin(["ciprofloxacin", "meropenem"])]
        .index.unique().astype(str)
    )
    
    # Read each file
    index_col = [
        "accession", "genus", "species",
        "antibiotic", "phenotype", "measurement_value"
    ]
    for file in files:
        data = pd.read_csv(
            file, sep="\t", compression="gzip", low_memory=False,
            index_col=index_col, dtype=str
        )
        
        # Filter columns depending on file type
        if "SNP" in file.name:
            valid = data.columns[
                data.columns.str.split("-").str.get(0).isin(aro)
            ]
        else:
            valid = data.columns[data.columns.isin(aro)]
        
        # Save filtered table
        data[valid].to_csv(
            f"{output}/{file.stem[:-4]}Biofiltered2.tsv.gz",
            compression="gzip",
            sep="\t"
        )

if __name__ == "__main__":
    main()
