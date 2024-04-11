This dataset contains 5,952 isolates with both WGS data and phenotypic data available. There are 8 species and for each one, the chosen antibiotic is either meropenem (as a representative of the carbapenem class) or ciprofloxacin (as a representative of the fluoroquinolone class). The chosen antibiotics are consistent between species and are based on the WHO's Priority Pathogens List (see Tacconelli et al, 2017 - DOI 10.1016/S1473-3099(17)30753-3). The columns in the dataset are as follows:

Taxonomic data:

genus, species - the taxonomic designation of the bacterial pathogen

Genotypic data:

accession - the accession number for WGS data; the majority (4,545 out of 5,952 isolates) have a WGS dataset available, the remainder are represented by an assembly

Phenotypic data:

phenotype - Resistant or Sensitive; this is what you will develop predictive models for in the main task; you may consider whether grouping genera and species is helpful
antibiotic - meropenem or ciprofloxacin
measurement_sign, measurement_value, measurement_unit - together these represent the MIC (minimum inhibitory concentration), which you will predict in the secondary task

Phenotypic metadata:

laboratory_typing_method - we only provided Broth dilution-based typing in this dataset and the testing dataset will also contain isolates phenotyped with Broth dilution
laboratory_typing_platform - this is not always specified, but may influence the interpretation of the MIC value as a binary phenotype (ie either Resistant or Sensitive)
testing_standard - we only provided CLSI entries
testing_standard_year - this is also not always specified, but may influence the interpretation of the MIC value as a binary phenotype (ie either Resistant or Sensitive)

Other metadata:

publication - when specified, this provides a PMID for the paper the entry is taken from
isolation_source, isolation_country - when specified, this provides information on where the isolate was taken from in the patient and where the patient comes from 
collection_date - when specified, this provides information on when the isolate was collected