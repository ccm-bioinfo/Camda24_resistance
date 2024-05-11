# Datasets description

### Training metadata dataset ([TrainingDataset.csv](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/aca2b4bef8c642f7ccd5adb63e7f054feb67783a/DataSets/TrainingDataset.csv))

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

### Testing metadata dataset ([TestingDataset.csv](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/d105fa8d90f8b3c397958d09d1fa615597883459/DataSets/TestingDataset.csv))

Metadata for the testing dataset. Includes the following columns: **genus**,
**species**, **accession**, **antibiotic** and **collection_date**.

### Resistance categories metadata ([ResistanceCategories.tsv](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/d105fa8d90f8b3c397958d09d1fa615597883459/DataSets/ResistanceCategories.tsv))

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

### Antibiotic Resistance Count Tables ([ResistanceCountFull.tsv](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/394c7fc8de1a265fe4911ad1335e1691efbb5637/DataSets/ResistanceCountFull.tsv) and [ResistanceCountStrict.tsv](https://raw.githubusercontent.com/ccm-bioinfo/Camda24_resistance/394c7fc8de1a265fe4911ad1335e1691efbb5637/DataSets/ResistanceCountStrict.tsv))

Two tables storing counts of genes conferring antibiotic resistance. They
contain the following columns:

- **accession**, **genus**, **species** - sample metadata.
- **phenotype**, **antibiotic**, **measurement_value** - prediction target.
- **all columns starting with 300...** - gene ARO ids (from
  [CARD](https://card.mcmaster.ca/)).

The **ResistanceCountFull.tsv** contains complete counts, even for genes with
low confidence. On the other hand, **ResistanceCountStrict.tsv** only includes
genes with a high confidence.