# BERT Fake News Detector

A **DistilBERT-based fake news detection system**.  
Users can train their own model and predict news as **REAL** or **FAKE**.

---

## Folder Structure

bert-fake-news-detector/
├── bert_fake_news.py # Training script
├── predict_fake_news.py # Prediction script
├── data/ # CSV datasets
│ ├── True.csv # Real news examples
│ └── Fake.csv # Fake news examples
├── results/ # Folder for trained model (created automatically)
├── README.md # This file
└── .gitignore # Ignores results/ and cache files

yaml
Copy code

**Note:** The `results/` folder is empty initially and will be created automatically after training.

---

## Installation

Recommended: use **Conda**:

```bash
conda create -n fake_news_venv python=3.10
conda activate fake_news_venv
pip install pandas scikit-learn torch transformers datasets
Training the Model
Place your CSV files in the data/ folder:

True.csv → REAL news

Fake.csv → FAKE news

Run the training script:

bash
Copy code
python bert_fake_news.py
Default: 100 samples per class, 2 epochs.

Trained model and tokenizer will be saved in results/.

Predicting News
Use the trained model in results/ with the prediction script:

bash
Copy code
python predict_fake_news.py
The script supports:

Predefined test examples

Interactive input (type news text to predict)

Output shows:

Predicted label: REAL news or FAKE news

Probabilities for each class

Notes
Increase samples and epochs in bert_fake_news.py for better accuracy.

results/ is created automatically and should not be tracked in Git.

Ensure the data/ folder exists with proper CSV files.

Optional
Modify scripts for batch predictions or custom paths.

Add your own datasets in data/ for training.

yaml
Copy code

---

After creating this `README.md` file:  

```bash
git add README.md
git commit -m "Add README with instructions and folder structure"
git push