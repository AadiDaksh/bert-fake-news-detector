import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import argparse

# User can control how many rows per class to train
parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=100, help="Samples per class (for quick testing)")
args = parser.parse_args()

#load dataset
print("Loading datasets...")
df_fake = pd.read_csv(os.path.join("fake_news_detector", "data", "Fake.csv"))
df_real = pd.read_csv(os.path.join("fake_news_detector", "data", "True.csv"))

# limit sample size (bigger n -> longer time to train model but more accurate)
if args.samples is not None:
    df_fake = df_fake.sample(n=min(args.samples, len(df_fake)), random_state=42)
    df_real = df_real.sample(n=min(args.samples, len(df_real)), random_state=42)


df_fake['label'] = 1  # FAKE
df_real['label'] = 0  # REAL

# Combine datasets
df = pd.concat([df_fake, df_real], ignore_index=True)
print("Total rows:", len(df))

#fill missing text/title
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
df['text_all'] = df['title'] + ' ' + df['text']

# split train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text_all'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# prepare hugging face dataset
train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
test_dataset = Dataset.from_dict({'text': test_texts.tolist(), 'label': test_labels.tolist()})

#tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

print("Tokenizing train dataset...")
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=64)
print("Tokenizing test dataset...")
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=64)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

#load model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,    
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    save_strategy="epoch",
    learning_rate=2e-5,
    report_to="none"                   # disables WandB logging
)

#Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


print("Training started...")
trainer.train()
print("Training finished!")

#Evaluation
print("Evaluating...")
preds = trainer.predict(test_dataset)
pred_labels = preds.predictions.argmax(-1)

print("\n===== Classification Report =====")
print(classification_report(test_labels, pred_labels))
print("\n===== Confusion Matrix =====")
print(confusion_matrix(test_labels, pred_labels))

# Save trained model and tokenizer
results_dir = os.path.join("fake_news_detector", "results")
os.makedirs(results_dir, exist_ok=True)

model.save_pretrained(results_dir)
tokenizer.save_pretrained(results_dir)
print(f"Trained model saved in {results_dir} folder!")


