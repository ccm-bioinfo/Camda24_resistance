# Datasets description

> [!NOTE]  
> Files ending with `.tsv` use tabulators, and files ending with `.csv` use
> commas, as separators. Some files are compressed (e.g. have `.gz` or `bz2`
> extension) to save space.

### 1. Training metadata dataset ([TrainingDataset.csv](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/TrainingDataset.csv))

This dataset contains 5,952 isolates with both WGS data and phenotypic data available. There are 8 species and for each one, the chosen antibiotic is either meropenem (as a representative of the carbapenem class) or ciprofloxacin (as a representative of the fluoroquinolone class). The chosen antibiotics are consistent between species and are based on the WHO's Priority Pathogens List (see Tacconelli et al, 2017 - DOI 10.1016/S1473-3099(17)30753-3). The columns in the dataset are as follows:

- Taxonomic data:
    - **genus**, **species** - the taxonomic designation of the bacterial pathogen
- Genotypic data:
    - **accession** - the accession number for WGS data; the majority (4,545 out of 5,952 isolates) have a WGS dataset available, the remainder are represented by an assembly
- Phenotypic data:
    - **phenotype** - Resistant or Sensitive; this is what you will develop predictive models for in the main task; you may consider whether grouping genera and species is helpful
    - **antibiotic** - meropenem or ciprofloxacin
    - **measurement_sign**, **measurement_value**, **measurement_unit** - together these represent the MIC (minimum inhibitory concentration), which you will predict in the secondary task
- Phenotypic metadata:
    - **laboratory_typing_method** - we only provided Broth dilution-based typing in this dataset and the testing dataset will also contain isolates phenotyped with Broth dilution
    - **laboratory_typing_platform** - this is not always specified, but may influence the interpretation of the MIC value as a binary phenotype (ie either Resistant or Sensitive)
    - **testing_standard** - we only provided CLSI entries
    - **testing_standard_year** - this is also not always specified, but may influence the interpretation of the MIC value as a binary phenotype (ie either Resistant or Sensitive)
- Other metadata:
    - **publication** - when specified, this provides a PMID for the paper the entry is taken from
    - **isolation_source**, **isolation_country** - when specified, this provides information on where the isolate was taken from in the patient and where the patient comes from 
    - **collection_date** - when specified, this provides information on when the isolate was collected

### 2. Testing metadata dataset ([TestingDataset.csv](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/TestingDataset.csv))

Metadata for the testing dataset. Includes the following columns: **genus**,
**species**, **accession**, **antibiotic** and **collection_date**.

### 3. Resistance categories metadata ([ResistanceCategories.tsv](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/ResistanceCategories.tsv))

Includes extra information regarding the ARO ids, i.e., unique numbers that
identify genes conferring antibiotic resistance from
[CARD](https://card.mcmaster.ca/). Contains the following columns:

- **ARO** - ARO id.
- **Gene Name** - common name for the gene.
- **Antibiotic** - name of the antibiotic for which the gene confers resistance.
  When a gene protects against multiple antibiotics, a comma-separated list is
  written instead. Can be empty if the antibiotic is unknown.
- **AMR Gene Family** - gene family which the gene belongs to. A gene can belong
  to multiple families, and a comma-separated list of families is written in
  such cases.
- **Drug Class** - type of substance that the gene produces. A gene can produce
  many types of substances, so a comma-separated list of substances is used in
  those cases.
- **Resistance Mechanism** - method which the gene uses to protect the organism
  against the antibiotic. Some genes have multiple resistance mechanisms, and,
  just like with previous columns, a comma-separated list of mechanisms is
  written for them.

### 4. Pangenome count tables (PangenomeCount*.tsv.bz2)

> [!IMPORTANT]  
> These files are compressed to save space. Decompress with `bunzip2 file`.

Four tables storing counts of all genes, based on a pangenome. Contains the
following columns:

- **accession**, **genus**, **species**, **antibiotic** - sample metadata.
- **phenotype**, **measurement_value** - prediction target.
- **everything else** - genes counts. Column labels refer to a gene family name.

Two tables contain the complete pangenome, divided by antibiotic:

- [**PangenomeCountCiprofloxacin.tsv.bz2**](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/PangenomeCountCiprofloxacin.tsv.bz2) - gene counts for samples tested against ciprofloxacin.
- [**PangenomeCountMeropenem.tsv.bz2**](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/PangenomeCountMeropenem.tsv.bz2) - gene counts for samples tested against meropenem.

And the other two tables store data of specific bacteria:

- [**PangenomeCountEscherichiaSalmonella.tsv.bz2**](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/PangenomeCountEscherichiaSalmonella.tsv.bz2) - gene counts for *Escherichia coli* and *Salmonella enterica* for both antibiotics.
- [**PangenomeCountPseudomonas.tsv.bz2**](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/PangenomeCountPseudomonas.tsv.bz2) - gene counts for *Pseudomonas aeruginosa* for both antibiotics.

> [!CAUTION]
> Some rows have empty values in the **phenotype** and **measurement_value**
> variables; these rows belong to the **testing dataset**.
> 
> Consequently, rows that do have these variables correspond to the
> **training dataset**.

### 5. Resistance gene count tables ([ResistanceGeneCountLoose.tsv.gz](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/ResistanceGeneCountLoose.tsv.gz) and [ResistanceGeneCountStrict.tsv.gz](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/ResistanceGeneCountStrict.tsv.gz))

> [!IMPORTANT]
> These files are compressed to save space. Decompress with `gunzip file`.

Two tables containing counts of AMR genes for different bacterial isolates.
Includes sample metadata (accession number, genus, species, phenotype,
antibiotic, and measurement value) and counts of AMR genes labeled with ARO ids. 

Both tables contain the following columns:

- **accession, genus, species, antibiotic** - sample metadata.
- **phenotype** - the first prediction target. Can have one of two values: Resistant or Susceptible.
- **measurement_value** - the second prediction target. Stores positive numeric values representing the MIC (minimum inhibitory concentration).
- **columns starting with 300...** - feature data. Contains counts of genes conferring antibiotic resistance. Columns are labeled using ARO ids (see Section 3 for details).

The difference between the two tables has to do with the confidence of the gene
presence. The **ResistanceGeneCountStrict.tsv.gz** file contains genes with a
high confidence, whereas **ResistanceGeneCountLoose.tsv.gz** includes genes with
low and high confidence, and is thus a superset of the first table.

> [!CAUTION]
> Some rows have empty values in the **phenotype** and **measurement_value**
> variables; these rows belong to the **testing dataset**.
> 
> Consequently, rows that do have these variables correspond to the
> **training dataset**.

### 6. Resistance SNP count tables ([ResistanceSNPCountLoose.tsv.gz](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/ResistanceSNPCountLoose.tsv.gz) and [ResistanceSNPCountStrict.tsv.gz](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/main/DataSets/ResistanceSNPCountStrict.tsv.gz))

> [!IMPORTANT]
> These files are compressed to save space. Decompress with `gunzip file`.

Two tables containing counts of AMR SNPs for different bacterial isolates.
Both tables contain the following columns:

- **accession, genus, species, antibiotic** - sample metadata.
- **phenotype** - the first prediction target. Can have one of two values: Resistant or Susceptible.
- **measurement_value** - the second prediction target. Stores positive numeric values representing the MIC (minimum inhibitory concentration).
- **columns starting with 300...** - feature data. Contains counts of SNPs conferring antibiotic resistance. Columns are labeled using ARO ids (see Section 3 for details) followed by the amino acid substitution.

The difference between the two tables has to do with the confidence of the gene
presence. The **ResistanceSNPCountStrict.tsv.gz** file contains genes with a
high confidence, whereas **ResistanceSNPCountLoose.tsv.gz** includes genes with
low and high confidence, and is thus a superset of the first table.

> [!CAUTION]
> Some rows have empty values in the **phenotype** and **measurement_value**
> variables; these rows belong to the **testing dataset**.
> 
> Consequently, rows that do have these variables correspond to the
> **training dataset**.
