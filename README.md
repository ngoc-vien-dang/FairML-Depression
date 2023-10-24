# Fairness and Bias Mitigation in ML-based Depression Prediction
Author: Vien Ngoc Dang

<p align="center">
  <img src="imgs/fairness_pipeline.png">
</p>

This is the implementation of the paper <b><i>Fairness and bias correction in machine learning for depression prediction: results from four different study populations</i></b> ([Dang et al., 2022](https://arxiv.org/abs/2211.05321)). In this project, we will see how to audit and address algorithmic bias. 

#### Supported bias mitigation strategies
* Suppression (SUP)
* Reweighing (RW) ([Kamiran and Calders, 2012](https://link.springer.com/article/10.1007/s10115-011-0463-8))
* Disparate Impact Remover (DIR) ([Feldman et al., 2015](https://dl.acm.org/doi/10.1145/2783258.2783311))
* Calibrated Equalized Odds Postprocessing (CPP) ([Pleiss et al., 2017](https://papers.nips.cc/paper/2017/hash/b8b9c74ac526fffbeb2d39ab038d1cd7-Abstract.html))
* Population Sensitivity-Guided Threshold Adjustment (PSTA)(our proposed post-hoc disparity mitigation method)

#### How to run the code
* ./utils and ./algorithms contain python files to implement ML models and bias mitigation algorithms
* The jypyter notebooks showed different steps to build an fair ML model associated with a *categorical* protected attribute (Notebook1) and a *continous* protected attribute (Notebook2)

#### Data availability
All four datasets are free to download for research purposes from:
- LONGSCAN: [https://www.ndacan.acf.hhs.gov/datasets/dataset-details.cfm?ID=170](https://www.ndacan.acf.hhs.gov/datasets/dataset-details.cfm?ID=170)
- FUUS: [https://datadryad.org/stash/dataset/doi:10.5061/dryad.54qt7](https://datadryad.org/stash/dataset/doi:10.5061/dryad.54qt7)
- NHANES: [https://wwwn.cdc.gov/nchs/nhanes/default.aspx](https://wwwn.cdc.gov/nchs/nhanes/default.aspx)
- UK Biobank (UKB): [https://www.ukbiobank.ac.uk/enable-your-research/register](https://www.ukbiobank.ac.uk/enable-your-research/register)

FUUS and NHANES datasets are open access and can be downloaded directly from their links above. LONGSCAN and UKB datasets can be accessed under request to the National Data Archive on Child Abuse and Neglect (NDACAN) and the UK Biobank Access Management Team, respectively. The results for UKB in this study will be returned to the UK Biobank within 6 months since publication, as required.


