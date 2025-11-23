BERT Fake News Detector

This project uses DistilBERT to detect whether news is REAL or FAKE. Users can train their own model and make predictions on new news text.

Project Structure

bert_fake_news.py – Script for training the model.

predict_fake_news.py – Script for predicting news.

data/ – CSV files for training (True.csv for real news, Fake.csv for fake news).

results/ – Folder where the trained model and tokenizer are saved (created automatically).

.gitignore – Excludes results/ and cache files.

README.md – Instructions.

Installation

Using Conda:

conda create -n fake_news_venv python=3.10
conda activate fake_news_venv
pip install pandas scikit-learn torch transformers datasets

Training

Place CSV files in data/.

Run:

python bert_fake_news.py


Default: 100 samples per class, 2 epochs.

Model and tokenizer saved in results/.

Predicting

Run:

python predict_fake_news.py


Supports predefined examples or interactive input.

Outputs predicted label (REAL or FAKE) and probabilities.

Notes

Increase samples/epochs for better accuracy.

Ensure data/ exists and has proper CSV files.

results/ is created automatically and is not tracked by Git.

Quick Start
git clone https://github.com/AadiDaksh/bert-fake-news-detector.git
cd bert-fake-news-detector
conda create -n fake_news_venv python=3.10
conda activate fake_news_venv
pip install pandas scikit-learn torch transformers datasets
python bert_fake_news.py
python predict_fake_news.py