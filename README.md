# Reliability-Aware Hybrid Quantum-Classical Diabetes Prediction

## Dataset
Place the original Pima Indians Diabetes CSV at `dataset/diabetes.csv`. Expected shape: 768 rows × 9 columns. Do not commit the dataset unless its license/terms permit redistribution.

## Notebook execution order
1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Preprocessing.ipynb`
3. `notebooks/03_Classical_ML.ipynb`
4. `notebooks/04_Feature_Selection.ipynb`
5. `notebooks/05_Quantum_Kernel_SVM.ipynb`
6. `notebooks/06_VQC.ipynb`
7. `notebooks/07_Experimental_Analysis.ipynb`

## Google Colab
Upload/commit this folder to GitHub. In Colab, open each notebook from GitHub. Add `dataset/diabetes.csv` to the repository only if permitted, or upload it to the Colab session / mount Drive. Notebook cells include basic Colab installation commands.

## Important research safeguards
- Split data with stratification.
- Fit imputers, scalers, feature selectors and threshold selection using training data only.
- Keep the final test set untouched until final evaluation.
- Quantum kernel computation is expensive; start with small, stratified training subsets and report the sample count.
- Do not claim quantum speedup or clinical validity.

## Current status
These notebooks are an implementation starter kit. Run sequentially, inspect outputs, and save artifacts/results. Some result-export paths may need adjustment depending on whether notebooks are opened from GitHub, Colab, or a local clone.
