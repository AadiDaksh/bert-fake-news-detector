# BERT Fake News Detector

A DistilBERT-based fake news detection system that can classify news articles as REAL or FAKE. Users can train their own model using example datasets and make predictions on new news text.

## Project Overview

This project contains two main scripts that work together to provide a complete fake news detection pipeline:

- **bert_fake_news.py** — Trains the model using CSV files of real and fake news
- **predict_fake_news.py** — Uses the trained model to predict whether a news article is real or fake

The trained model and tokenizer are saved automatically in a `results` folder after training.

## Folder Structure

```
├── data/
│   ├── True.csv          # Real news examples
│   └── Fake.csv          # Fake news examples
├── results/              # Trained model and tokenizer (created automatically)
├── .gitignore           # Ensures results/ and cache files are not tracked
├── README.md            # This file
├── bert_fake_news.py    # Training script
└── predict_fake_news.py # Prediction script
```

## Installation

It is recommended to use a Conda environment:

### Step 1: Create and Activate Environment
```bash
conda create -n fake_news_venv python=3.10
conda activate fake_news_venv
```

### Step 2: Install Dependencies
```bash
pip install pandas scikit-learn torch transformers datasets
```

## Usage

### Training the Model

1. Place your CSV files in the `data/` folder:
   - `True.csv` — Real news examples
   - `Fake.csv` — Fake news examples

2. Run the training script:
```bash
python bert_fake_news.py
```

**Default settings:**
- 100 samples per class
- 2 training epochs

The trained model and tokenizer will be saved in the `results/` folder.

### Making Predictions

Use the trained model to make predictions on new news articles:

```bash
python predict_fake_news.py
```

**The script allows you to:**
- Use predefined test examples
- Input custom news text interactively

**Output includes:**
- Predicted label: **REAL** or **FAKE** news
- Confidence probabilities for each class

## Tips & Notes

- **Improve accuracy** by increasing the number of samples and training epochs in `bert_fake_news.py`
- **Ensure proper data format** — Make sure the `data/` folder exists with properly formatted CSV files
- **Git tracking** — The `results/` folder is automatically created and should not be tracked by Git (see `.gitignore`)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/AadiDaksh/bert-fake-news-detector.git
cd bert-fake-news-detector
```

2. Set up Conda environment and install dependencies:
```bash
conda create -n fake_news_venv python=3.10
conda activate fake_news_venv
pip install pandas scikit-learn torch transformers datasets
```

3. Train the model:
```bash
python bert_fake_news.py
```

4. Make predictions:
```bash
python predict_fake_news.py
```

## Requirements

- Python 3.10+
- pandas
- scikit-learn
- torch
- transformers
- datasets

## License

This project is open source and available for educational and research purposes.
