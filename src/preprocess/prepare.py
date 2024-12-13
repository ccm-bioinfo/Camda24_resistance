"""load in and prepare data from genomics processing pipeline."""

import pandas as pd

ANTIBIOTICS = ["Ciprofloxacin", "Meropenem"]

GENOMIC_SCAN_TYPE = "Strict"
data_dir = "./../../../Datasets/"

def combine_data(antibiotics=ANTIBIOTICS, genomic_scan_type=GENOMIC_SCAN_TYPE):
    """ Combine data from genomic scan for multiple antibiotics.
    Args:
        antibiotics: List of antibiotics to combine data for.
        genomic_scan_type: Type of genomic scan to combine data for.
    Returns:
        combined_df: DataFrame containing combined data for all antibiotics.
    """
    # Load data for each antibiotic
    dfs = []
    for antibiotic in antibiotics:
        file_path = f"{data_dir}Resistance{antibiotic}{genomic_scan_type}.tsv.gz"
        df = pd.read_csv(file_path, sep="\t", compression="gzip", low_memory=False) # low_memory=False to suppress warning of mixed types
        df['antibiotic'] = antibiotic
        dfs.append(df)
    
    # Combine data for all antibiotics
    combined_df = pd.concat(dfs)
    combined_df.to_csv(f"{data_dir}combined_antibiotic_resistance_{genomic_scan_type.lower()}.tsv.gz", index=False, compression="gzip", sep="\t")
    print("Size of combined data:", combined_df.shape)
    print("Save location combined data:", f"{data_dir}combined_antibiotic_resistance_{genomic_scan_type.lower()}.tsv.gz")
    return combined_df

if __name__ == "__main__":
    combined_df = combine_data()
    print(combined_df.head())