# Preprocessing scripts

All scripts require Python and Pandas, and were tested with Python 3.12.3 and
Pandas 2.2.2.

> [!CAUTION]
> All scripts use the information stored in the training and testing metadata
> tables (see DataSets readme). If the accessions do not coincide, they will not
> be included in the output table. This means that:
> - Files passed to `amr-count.py` must be named ***accession***.tsv.
> - The matrix.csv file passed to `pangenome-count.py` must have its genome
> columns labeled as ***accession***.
> 
> Where ***accession*** is a valid accession taken from the metadata tables.

### Resistance count table generator ([amr-count.py](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/66de738d206c145975ff1f5f551bce99598675d5/Scripts/preprocessing/amr-count.py))

usage: `amr-count.py [-h] [-s] file [file ...]`

Print a gene count TSV table from RGI main tab delimited files.

positional arguments:
- `file` - RGI main tab delimited file

options:
- `-h`, `--help` - show this help message and exit.
- `-s`, `--strict` - do not include loose hits.

Example usage: `./amr-count.py *.tsv > output.tsv`

### Pangenome count table generator ([pangenome-count.py](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/7aa1f7f331a27e8228adf2d9a472da54dc9ec5a1/Scripts/preprocessing/pangenome-count.py))

> [!CAUTION]  
> Uses up to 32GiB of RAM

usage: `pangenome-count.py [-h] -a {ciprofloxacin,meropenem} file`

Print a gene count TSV table from a PPanGGOLiN matrix.csv file.

positional arguments:
- `file` - PPanGGOLiN matrix.csv file

options:
- `-h`, `--help` - show this help message and exit
- `-a {ciprofloxacin,meropenem}`, `--antibiotic {ciprofloxacin,meropenem}` - 
  type of antibiotic

Example usage: `./pangenome-count.py -a meropenem matrix.csv > output.tsv`
