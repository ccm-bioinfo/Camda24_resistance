# Preprocessing scripts

All scripts require Python and Pandas, and were tested with Python 3.12.3 and
Pandas 2.2.2.

### Resistance count table generator ([amr-count.py](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/66de738d206c145975ff1f5f551bce99598675d5/Scripts/preprocessing/amr-count.py))

usage: `amr-count.py [-h] [-s] file [file ...]`

Print a gene count TSV table from RGI main tab delimited files.

positional arguments:
- `file` - RGI main tab delimited file

options:
- `-h`, `--help` - show this help message and exit.
- `-s`, `--strict` - do not include loose hits.

Example usage: `./amr-count.py *.tsv > output.tsv`

### Pangenome count table generator ([pangenome-count.py](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/66de738d206c145975ff1f5f551bce99598675d5/Scripts/preprocessing/pangenome-count.py))

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
