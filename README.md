<div align="center">
<img align="center" width="30%" alt="image" src="https://www.sandiego.edu/assets/global/images/logos/logo-usd.png">
</div>

# Music Genre and Composer Classification Using Deep Learning

![](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![](https://img.shields.io/badge/MSAAI-DL-blue?style=for-the-badge)
![](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)

aai-511-group6-final-project

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FPSswathi%2Fmusic-genre-and-composer-classification%2Fvs-badge&countColor=%23263759)

## Table of Contents
- [Description](#Overview)
- [Results](#Results)
- [Usage](#usage)
- [Contributing](#Contributors)
- [License](#license)

#### Github Project Structure

```
MUSIC-GENRE-AND-COMPOSER-CLASSIFICATION/
│
├── featured datasets/                  # Datasets used for training & evaluation
│   ├── data_splits.npz                  # Preprocessed train/val/test splits
│   └── datasets.zip                     # Raw MIDI dataset archive
│
├── models/                              # Saved trained model weights
│   ├── cnn_piano_roll.keras              # Final CNN model (piano roll input)
│   ├── final_bi_lstm_attention_model.keras # Final BiLSTM + Attention model
│   └── final_lstm_model.keras            # Final LSTM model
│
├── notebooks/                           # Jupyter notebooks for experiments
│   ├── helpers/
│   │   └── processor.py                  # Preprocessing utilities (feature extraction, augmentation)
│   ├── cnn_piano_roll.ipynb              # CNN training & evaluation pipeline
│   └── notelevel-lstm-bilstm-attn-swathi.ipynb  # LSTM & BiLSTM + Attention training & evaluation
|    ├── merged-compared-lstm-bilstm-cnn.ipynb # Combined metrics & comparison notebook
|
│
├── proposal/                            # Project proposal & documentation
│   └── AAI-511-group6 Final Team Project Status Update For…pdf
│
├── selectedcomposers/                   # Selected composer MIDI datasets
│   ├── Bach/
│   ├── Beethoven/
│   ├── Chopin/
│   └── Mozart/
│
│
├── environment.yml                      # Conda environment file with dependencies
├── Makefile                              # Automation commands (optional build/run targets)
├── README.md                             # Project documentation (this file)
└── .gitignore                            # Ignored files for Git
```

# Overview
## About the Project

Music is a universal language that transcends cultures and generations. Classical composers such as Bach, Beethoven, Mozart, and Chopin are known for their distinct styles, yet their compositions often share structural and melodic similarities that make it difficult—even for trained musicians to distinguish between them by ear or by score alone.

This project aims to explore whether computational techniques, specifically deep learning, can learn to distinguish between these composers based on their musical characteristics. By analyzing patterns in musical structure, dynamics, tempo, pitch, and harmony extracted from MIDI files, we seek to develop a system that can predict the composer of a given musical piece.

Using a dataset of MIDI files collected from classical composers, we preprocess the data by reshaping the midi files to arrays of piano roll sequences (piano roll approach), or extracting chunks of 200 notes and computing a range of statistical and musical features (extracted features approach). We then train deep learning models—including Long Short-Term Memory (LSTM) networks, BiDirectional LSTMS with an Attention layer, and Convolutional Neural Networks (CNNs) to learn and classify the compositional styles of the selected composers.

## Dataset Description

The <a href = "https://www.kaggle.com/datasets/blanderbuss/midi-classic-music/">dataset</a> used in this project consists of MIDI files representing classical music compositions by four renowned composers: Johann Sebastian Bach, Ludwig van Beethoven, Frédéric Chopin, and Wolfgang Amadeus Mozart. The dataset was sourced from a publicly available collection on Kaggle, containing hundreds of MIDI files organized by composer.

The data was handled differently for CNNs and for the LSTMs. For the CNNs we preprocessed the data by reshaping the midi files to arrays of piano roll sequences (piano roll approach). While for the LSTMs, musical features were extracted from the midi files. The midi files were afterwards split into chunks of 200 notes each to be used as inputs in the corresponding models. Each chunk kept its corresponding set of locally computed features. The Exploratory Data Analysis was conducted on the extracted features dataset before chunking.

After filtering for the selected composers and segmenting each piece into chunks of 200 notes, the final dataset contains approximately 23,674 rows (or segments), with each row representing a distinct 200-note chunk for the extracted feature approach. Each row includes 18 variables, such as composer label, filename, and 12 numerical musical features like tempo, average pitch, pitch range, note duration statistics, velocity, and chord density. The dataset is approximately 40–50 MB in memory size after processing and provides a rich foundation for learning stylistic differences between composers through statistical and temporal patterns in music.

For the piano roll after unwrapping the sequence data, we ended with about 750k rows, where ~300k rows of data belong to Bach, ~200k to Beethoven and Mozart (each), and 57k from Chopin. Given the different shapes, lengths, and class distributions, we implemented distinct data preprocessing techniques for each of these approaches.

## Project Considerations

From both composer-averaged metrics and composer-level metrics, we were able to extract some meaningful insights. Overall, the models performed worse on recall when compared to precision, and accuracy tends to be closer to  precision. Mozart was challenging for the models to identify. It mistook a lot of pieces belonging to Mozart as belonging to other composers. 

MIDI files were preprocessed to extract features such as tempo, average pitch, pitch variance, chord density, and note durations using the pretty_midi python library. To train the models we tested locally on our machines smaller versions of the datasets and moved on to the cloud for the more computationally expensive tasks.

# Results

**Overall Performance Comparison Table:**

| **Metric**               | **LSTM Model** | **BiLSTM + Attention Model**  | **CNN**|
|--------------------------|----------------|-------------------------------|--------|
| **Accuracy**             | 85.06%         | 90.38%                        | 78.6%  |
| **Precision**            | 0.85           | 0.90                          | 0.84   |
| **Recall**               | 0.89           | 0.93                          | 0.68   |
| **F1-Score**             | 0.86           | 0.91                          | 0.75   |

Training Deep Learning architectures on MIDI files proved to be feasible and effective. Leveraging the ability of DL to process data representations in the form of tensors, we were able to achieve composer prediction accuracies ranging from 0.78 to 0.92. The LSTM, BiLSTM, CNN models demonstrated strong capability in composer classification using MIDI-derived features. The BiLSTM achieved the highest accuracy and stability , followed by the LSTM. Making them more reliable choices for this task than our trained CNN model.

# Usage

## Prerequisites: 

### Python Version
For this project we are using Python version 3.11, conda automatically will install and set the correct python version for the project so there is nothing that needs to be done.

### 1. Install Miniconda

If you are already using Anaconda or any other conda distribution, feel free to skip this step.

Miniconda is a minimal installer for `conda`, which we will use for managing environments and dependencies in this project. Follow these steps to install Miniconda or go [here](https://docs.anaconda.com/miniconda/install/) to reference the documentation: 

1. Open your terminal and run the following commands:
```bash
   $ mkdir -p ~/miniconda3

   <!-- If using Apple Silicon chip M1/M2/M3 -->
   $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
   <!-- If using intel chip -->
   $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda3/miniconda.sh

   $ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   $ rm ~/miniconda3/miniconda.sh
```

2. After installing and removing the installer, refresh your terminal by either closing and reopening or running the following command.
```bash
$ source ~/miniconda3/bin/activate
```

3. Initialize conda on all available shells.
```bash
$ conda init --all
```

You know conda is installed and working if you see (base) in your terminal. Next, we want to actually use the correct environments and packages.

### 2. Install Make

Make is a build automation tool that executes commands defined in a Makefile to streamline tasks like compiling code, setting up environments, and running scripts. [more information here](https://formulae.brew.sh/formula/make)

#### Installation

`make` is often pre-installed on Unix-based systems (macOS and Linux). To check if it's installed, open a terminal and type:
```bash
make -v
```

If it is not installed, simply use brew:
```bash
$ brew install make
```

#### Available Commands

The following commands are available in this project’s `Makefile`:

- **Set up the environment**:

    This will create the environment from the environment.yml file in the root directory of the project.

    ```bash
      $ make create
    ```

- **Update the environment**:

    This will update the environment from the environment.yml file in the root directory of the project. Useful if pulling in new changes that have updated the environment.yml file.

    ```bash
      $ make update
    ```

- **Remove the environment**:

    This will remove the environment from your shell. You will need to recreate and reinstall the environment with the setup command above.

    ```bash
    $ make clean
    ```

- **Activate the environment**:

    This will activate the environment in your shell. Keep in mind that make will not be able to actually activate the environment, this command will just tell you what conda command you need to run in order to start the environment.

    Please make sure to activate the environment before you start any development, we want to ensure that all packages that we use are the same for each of us.

    ```bash
    $ make activate
    ```

    Command you actually need to run in your terminal:
    ```bash
    $ conda activate composer-env
    ```

- **Deactivate the environment**:

    This will Deactivate the environment in your shell.

    ```bash
    $ make deactivate
    ```

- **run jupyter notebook**:

    This command will run jupyter notebook from within the conda environment. This is important so that we can make sure the package versions are the same for all of us! Please make sure that you have activated your environment before you run the notebook.

    ```bash
    $ make notebook
    ```

- **Export packages to env file**:

    This command will export any packages you install with either `conda install ` or `pip install` to the environment.yml file. This is important because if you add any packages we want to make sure that everyones machine knows to install it.

    ```bash
    $ make freeze
    ```

- **Verify conda environment**:

    This command will list all of your conda envs, the environment with the asterick next to it is the currently activated one. Ensure it is correct.

    ```bash
    $ make verify
    ```

#### Example workflows:

To simplify knowing which commands you need to run and when you can follow these instructions:

- **First time running, no env installed**:

    In the scenario where you just cloned this repo, or this is your first time using conda. These are the commands you will run to set up your environment.

    ```bash
    <!-- Make sure that conda is initialized -->
    $ conda init --all

    <!-- Next create the env from the env file in the root directory. -->
    $ make create

    <!-- After the environment was successfully created, activate the environment. -->
    $ conda activate composer-env 

    <!-- verify the conda environment -->
    $ make verify

    <!-- verify the python version you are using. This should automatically be updated to the correct version 3.11 when you enter the environment. -->
    $ python --version

    <!-- Run jupyter notebook and have some fun! -->
    $ make notebook
    ```

- **Installing a new package**:

    While we are developing, we are going to need to install certain packages that we can utilize. Here is a sample workflow for installing packages. The first thing we do is verify the conda environment we are in to ensure that only the required packages get saved to the environment. We do not want to save all of the python packages that are saved onto our system to the `environment.yml` file. 

    Another thing to note is that if the package is not found in the conda distribution of packages you will get a `PackagesNotFoundError`. This is okay, just use pip instead of conda to install that specific package. Conda thankfully adds them to the environment properly.

    ```bash
    <!-- verify the conda environment -->
    $ make verify

    <!-- Install the package using conda -->
    $ conda install <package_name>

    <!-- If the package is not found in the conda channels, install the package with pip. -->
    $ pip install <package_name>

    <!-- If removing a package. -->
    $ conda remove <package_name>
    $ pip remove <package_name>

    <!-- Export the package names and versions that you downloaded to the environment.yml file -->
    make freeze
    ```

- **Daily commands to run before starting development**:

    Here is a sample workflow for the commands to run before starting development on any given day. We want to first pull all the changes from github into our local repository, 

    ```brew
    <!-- Pull changes from git -->
    $ git pull origin main

    <!-- Update env based off of the env file. It is best to deactivate the conda env before you do this step-->
    $ conda deactivate
    $ make update
    $ conda activate composer-env 

    $ make notebook
    ```

- **Daily commands to run after finishing development**:

    Here is a sample workflow for the commands to run after finishing development for any given day.

    ```brew
    $ conda deactivate

    <!-- If you updated any of the existing packages, freeze to the environment.yml file. -->
    $ make freeze

    <!-- Commit changes to git -->
    $ git add .
    $ git commit -m "This is my commit message!"
    $ git push origin <branch_name>
    ```

## Setup & Installation

### Clone the repository

git clone https://github.com/yourusername/project-name.git

cd project-name

### Create virtual environment and install dependencies

## For Conda environment

### Setup
1. Create the environmnet, install packages
```
conda env create -f environment.yml
```
2. Activate the environment
```
conda activate composer-env
or
source activate composer-env
```

## For python environment

### Setup

python -m venv venv

source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

## Contributors
<table>
  <tr>
    <td>
        <a href="https://github.com/carlosOrtizM.png">
          <img src="https://github.com/carlosOrtizM.png" width="100" height="100" alt="Carlos Ortiz "/><br />
          <sub><b>Carlos Ortiz</b></sub>
        </a>
      </td>
      <td>
        <a href="https://github.com/PSswathi.png">
          <img src="https://github.com/PSswathi.png" width="100" height="100" alt="Swati Pabb "/><br />
          <sub><b>Swati Pabb</b></sub>
        </a>
      </td>
     <td>
      <a href="https://github.com/omarsagoo.png">
        <img src="https://github.com/omarsagoo.png" width="100" height="100" alt="Omar Sagoo"/><br />
        <sub><b>Omar Sagoo</b></sub>
      </a>
    </td>
  </tr>
</table>

## License

MIT License

Copyright (c) [2025]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.