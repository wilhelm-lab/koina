# DOME Questionnaire

This questionnaire is meant to help you in structuring the documentation of models published on Koina. It is based on the DOME standards.

DOME: recommendations for supervised machine learning validation in biology. 
Walsh, I., Fishman, D., Garcia-Gasulla, D. et al. 
Nat Methods 18, 1122–1127 (2021). https://doi.org/10.1038/s41592-021-01205-4

Interpretation of the DOME Recommendations for Machine Learning in Proteomics and MetabolomicsArticle link copied!
Palmblad, M., Böcker, S., Degroeve, S., et al.
J. Proteome Res. 2022, 21, 4, 1204–1207. https://doi.org/10.1021/acs.jproteome.1c00900


## Data
- **What is the source of the data used to train the model?**  
- **What type of molecules does the training data represent?**  
  (e.g., tryptic peptides, lipids, all metabolites)  
- **How well does the training data represent the complexity of the molecular class the model is intended for?**  
- **Was the data acquired on similar instruments/settings or across diverse conditions?**  
- **What are the limitations of the training data?**  
  (e.g., trained on tryptic peptides but intended for other peptides)  
- **Were there known false positives/negatives in the training data?**  
  How might they affect performance?  
- **Were there instrument performance issues during data acquisition?**  
  How might they impact test-set performance?  

## Optimization
- **What is the optimization target?**  
  (e.g., spectrum-, peptide-, or protein-level statistics)  
- **How does claimed performance compare to experimental variability?**  
  (e.g., peak intensity or retention time variability)  
- **What metric is used for chromatogram/spectra comparison?**  
  - Spectral angle, cosine score, or dot product?  
  - Were peaks discarded? What tolerances used? How were ambiguities resolved?  

## Model
- **Is the model interpretable or a black box?**  
- **If interpretable, how can it be interpreted?**  
  What insights does it provide?  
- **Is the model classification or regression?**  
- **How long does a single prediction take?**  
- **What are the model’s limitations?**  
  (e.g., chromatography conditions for RT prediction, ionization/fragmentation modes for spectra simulation)  

## Evaluation
- **What performance measures are reported?**  
- **Why were these measures chosen?**  
- **If reporting a single performance number, how is it justified?**  
- **Is a confusion matrix available?**  
- **Was performance compared to simpler baseline methods?**  
- **How was the model evaluated?**  
  (e.g., cross-validation, independent datasets, novel experiments)  
- **Has it been tested on independent data from different instruments/labs?**  
- **For RT prediction: What improvement in ID/quantification does it imply?**  
- **Does the model use input from other ML algorithms?**  
  If yes, specify which ones.  
- **Were precursor ion peaks excluded during spectra comparison?**  
- **Is performance highly variable?**
- **Applicability scope:** What data types work, and which don’t?  
- **Known performance issues users should be aware of?**  

## Input Notes
- **Input limitations:**  
  Supported PTMs, sequence length, CE range, charge states, etc.  

## Output Notes
- **Output considerations:**  
  Intensity normalization, fragment ion annotation, etc.  
