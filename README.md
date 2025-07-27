# aai-511-group6-final-project

### Music Genre and Composer Classification Using Deep Learning 

### Team Members:

Carlos A. Ortiz Montes De Oca

Omar Sagoo

Swathi Subramanyam Pabbathi

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

### Project Description

Project Description

Music is a universal language that transcends cultures and generations. Classical composers such as Bach, Beethoven, Mozart, and Chopin are known for their distinct styles, yet their compositions often share structural and melodic similarities that make it difficult—even for trained musicians to distinguish between them by ear or by score alone.

This project aims to explore whether computational techniques, specifically deep learning, can learn to distinguish between these composers based on their musical characteristics. By analyzing patterns in musical structure, dynamics, tempo, pitch, and harmony extracted from MIDI files, we seek to develop a system that can predict the composer of a given musical piece.

Using a dataset of MIDI files collected from classical composers, we preprocess the data by extracting chunks of 200 notes and computing a range of statistical and musical features. We then train deep learning models—including Long Short-Term Memory (LSTM) networks and Convolutional Neural Networks (CNNs) to learn and classify the compositional styles of the selected composers.

### Project Objective

The primary objective of this project is to develop a deep learning model capable of accurately predicting the composer of a given piece of music based on musical features extracted from MIDI files.

Specifically, we aim to:

1.	Build a preprocessed dataset by selecting MIDI files from four classical composers: Bach, Beethoven, Chopin, and Mozart.

2.	Segment and extract features from MIDI chunks using music analysis tools like pretty_midi. These features include tempo, pitch statistics, chord density, note duration, velocity, and more.

3.	Perform extensive exploratory data analysis (EDA) to understand the distributions, correlations, and differences in musical styles across composers.

4.	Train and evaluate two deep learning architectures:

    *	LSTM: To model temporal dependencies and sequential patterns.

    *   CNN: To capture local relationships and feature patterns.

5.	Assess model performance using accuracy, confusion matrix, and classification metrics, and interpret the model’s ability to distinguish between composers.

6.	Explore advanced techniques such as PCA, outlier handling, class balancing, and attention mechanisms to enhance model accuracy and interpretability.

By the end of this project, the developed model can serve as a tool for music students, historians, and enthusiasts to classify music by composer and better understand stylistic patterns in classical music.

### Dataset Description:

The dataset used in this project consists of MIDI files representing classical music compositions by four renowned composers: Johann Sebastian Bach, Ludwig van Beethoven, Frédéric Chopin, and Wolfgang Amadeus Mozart. The dataset was sourced from a publicly available collection on Kaggle, containing hundreds of MIDI files organized by composer.

After filtering for the selected composers and segmenting each piece into chunks of 200 notes, the final dataset contains approximately 23,674 rows (or segments), with each row representing a distinct 200-note chunk. Each row includes 15 variables, such as composer label, filename, and 12 numerical musical features like tempo, average pitch, pitch range, note duration statistics, velocity, and chord density. The dataset is approximately 40–50 MB in memory size after processing and provides a rich foundation for learning stylistic differences between composers through statistical and temporal patterns in music.
