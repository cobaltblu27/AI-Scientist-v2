# Title
Predicting Soybean Metabolite Traits from SNP Data and Reference Genome Information

# Keywords
soybean, SNP, reference genome, genomics, metabolite prediction, isoflavones

# TL;DR
We want to explore models that predict multiple soybean metabolite traits from SNP data across 333 cultivars, while also allowing the use of the soybean reference genome sequence as additional information.

# Abstract
This project focuses on predicting multiple metabolite-related traits in soybean using genomic data. The input includes SNP data for 333 soybean cultivars from `SNP_data.tsv.gz`, target metabolite values from `soybean_label_map_dedup.csv`, and a soybean reference genome sequence from `Gmax_275_v2.0.softmasked_filtered.fa.gz`.

The prediction targets include `daidzein_values`, `glycitein_values`, `genistein_values`, `glycitin`, `malonyl-daidzin_values`, `Malonyl-glycitin_values`, `Malonyl-genistin_values`, `daidzin_value`, and `genistin_value`.

The goal is to explore whether SNP information alone, or SNP information together with reference genome sequence information, can improve prediction of these metabolite traits. The project is open to a wide range of modeling directions, including methods that use genomic variation, sequence context, or relationships among traits.