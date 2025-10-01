# Bayesian Constraints on Pre-Equilibrium Jet Quenching and Predictions for Oxygen Collisions

This repository provides the data and plotting scripts associated with our paper:  
[**arXiv:2509.19430**](https://arxiv.org/abs/2509.19430)  

## Repository Structure

- **`data/`** – Numerical results with descriptive file names.  
- **`src/`** – Implementation of the **JETSCALE/STAT** framework, including the Gaussian Process (GP) emulator used to evaluate the quenching model across parameter space.  
- **`cache/`** – Stored GP emulator parameters.  
- **`plot_*.py`** – Plotting scripts to reproduce the main results from the paper.  
- **`figs/`** – Generated figures.  

## How to Use

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

<!-- =================================================================== -->
## JETSCAPE/STAT package info

### Links
repository home: [JETSCAPE/STAT](https://github.com/JETSCAPE/STAT/tree/master)

File format specification: [v1.0](https://www.evernote.com/l/ACWFCWrEcPxHPJ3_P0zUT74nuasCoL_DBmY)

File reader documentation: [link](https://www.evernote.com/l/ACXYRePvf2lNirII32b25Wg93rqD0kH1LSs)

[comment]: # (Previous documentation: http://hic-param-est.readthedocs.io/en/latest/ )

[comment]: # (-- Need to double check if everything up to date)

### Introduction

This is the package for statistical analysis

### Installation

1. Install python3, with packages `emcee` (<=2.2.1), `h5py`, `hic`, `numpy`, `PyYAML`, `scikit-learn` (>= 0.18), `scipy`, `pandas`, `pathlib`, `hsluv`, `matplotlib`.  Use pip to install them if needed

2. If you don't have R, download R from [here](https://cran.cnr.berkeley.edu/)

3. Open an R Console instance by opening the R app or by typing R in the command line.

4. In the R console, type the command `install.packages('lhs')` and pick an appropriate download mirror if prompted. To ensure the package was properly installed, type `library(lhs)` in the R console. If that command runs without error, the package is installed. Close the R console by typing `quit()`.

5. Clone this git repository

6. Open Terminal (OSX, Linux) or Windows Command Prompt (Windows).

7. Navigate to the downloaded/cloned git repository.

8. Type: `jupyter notebook`. This will open Jupyter iPython Notebook in a web browser.
In Jupyter, open `Example.ipynb`, and run the first cell. If it runs without error, then you should be properly set up for the program.

9. Execute all cells for an example of an analysis



