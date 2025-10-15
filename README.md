# Environment Setup for PIGNN Project

## Overview

This document provides instructions for setting up the development environment for the PIGNN (Physics-informed Neural Networks) project. The environment includes PyTorch with CUDA support, PyTorch Geometric, and essential data science and chemistry libraries.

## Prerequisites

* Conda package manager (Miniconda or Anaconda)
* NVIDIA GPU with CUDA 12.1 support (recommended)
* Windows

## Installation

### Option 1: One-Command Setup

Copy and paste this single command to install everything automatically:

```bash
conda create -n pignn_env python=3.11.9 -y && conda activate pignn_env && conda install pytorch=2.2.2 pytorch-cuda=12.1 torchvision=0.17.2 torchaudio=2.2.2 -c pytorch -c nvidia -y && pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html && pip install torch-geometric==2.5.3 && conda install numpy==1.26.3 pandas==2.2.2 matplotlib=3.8.4 seaborn=0.13.2 scikit-learn=1.7.1 networkx=3.2.1 rdkit=2023.09.5 jupyter ipykernel -c conda-forge -y && pip install captum==0.8.0 py3Dmol==2.4.2 plotly==6.2.0 kaleido==1.1.0 optuna==4.2.1 papermill
```

### Option 2: Step-by-Step Installation

If you prefer to run commands individually:

1. Create conda environment

```bash
conda create -n pignn_env python=3.11.9 -y
```

2. Activate environment

```bash
conda activate pignn_env
```

3. Install PyTorch with CUDA

```bash
conda install pytorch=2.2.2 pytorch-cuda=12.1 torchvision=0.17.2 torchaudio=2.2.2 -c pytorch -c nvidia -y
```

4. Install PyTorch Geometric dependencies

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

5. Install PyTorch Geometric

```bash
pip install torch-geometric==2.5.3
```

6. Install Data Science and Chemistry Packages

```bash
conda install numpy==1.26.3 pandas==2.2.2 matplotlib==3.8.4 seaborn==0.13.2 plotly==6.2.0 scikit-learn==1.7.1 networkx==3.2.1 rdkit==2023.09.5 jupyter ipykernel -c conda-forge -y
```

7. Install Data Science and Chemistry Packages

```bash
pip install captum==0.8.0 py3Dmol==2.4.2 plotly==6.2.0 kaleido==1.1.0 optuna==4.2.1 papermill
```

### Using Custom Python Modules

The project contains a folder `modules/` with multiple Python files. To make Python recognize these modules, follow the steps below:

**1. Locate your project folder**

Open the folder where you cloned the repository. Inside, find the `modules` folder. Note the full path (e.g., `E:\Virtual lab\amine_gnn\pignn\modules`).

**2. Open Environment Variables Settings**

* Press `Win + R`, type `sysdm.cpl`, and press Enter.
* Go to the **Advanced** tab → click **Environment Variables…**

**3. Add a new PYTHONPATH variable**

* Under **User variables**, click **New…**
* **Variable name:** `PYTHONPATH`
* **Variable value:** full path to your modules folder, e.g.,
  ```
  E:\Virtual lab\amine_gnn\pignn\modules
  ```
* Click **OK** to save.

**4. Update existing PYTHONPATH (if it exists)**

* If `PYTHONPATH` already exists, select it → click **Edit…**
* Add a semicolon `;` at the end of the current value, then add your modules path.

  Example:

  ```
  C:\some\other\path;E:\Virtual lab\amine_gnn\pignn\modules
  ```

**Test in Python**

Open you preferred IDE and run:

```
import sys
print(sys.path)
from EnhancedDataSplit import DataSplitter
```

All `.py` files in the `modules/` folder are automatically accessible after setting `PYTHONPATH`.

### Check if the installation is correct

Run the 'dependencies_test.py' at your installed conda environment (pignn_env)

## Package Versions

The environment includes the following specific versions:

| Package           | Version   | Purpose                    |
| ----------------- | --------- | -------------------------- |
| Python            | 3.11.9    | Base programming language  |
| PyTorch           | 2.2.2     | Deep learning framework    |
| PyTorch CUDA      | 12.1      | GPU acceleration           |
| TorchVision       | 0.17.2    | Computer vision utilities  |
| TorchAudio        | 2.2.2     | Audio processing utilities |
| PyTorch Geometric | 2.5.3     | Graph neural networks      |
| NumPy             | 1.26.3    | Numerical computing        |
| Pandas            | 2.2.2     | Data manipulation          |
| Matplotlib        | 3.8.4     | Data visualization         |
| Seaborn           | 0.13.2    | Statistical visualization  |
| Scikit-learn      | 1.7.1     | Machine learning utilities |
| NetworkX          | 3.2.1     | Graph analysis             |
| RDKit             | 2023.09.5 | Cheminformatics            |
| Jupyter           | -         | Interactive notebooks      |
| IPykernel         | -         | Jupyter kernel             |
| Captum            | 0.8.0     | Model interpretability     |
| py3Dmol           | 2.4.2     | 3D molecular visualization |
