import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default=os.path.join("fake_news_detector", "results"),
                    help="Path to the trained model and tokenizer")
args = parser.parse_args()

tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)

model.eval()  # set model to evaluation mode


# =========================
# 2️⃣ Prediction function
# =========================
def predict_news(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    label = "REAL news" if predicted_class_id == 0 else "FAKE news"
    return label, probs


# =========================
# 3️⃣ Test examples
# =========================
examples = [
    "Breaking news: Scientists discover a new method to convert plastic into fuel.",
    "Celebrity announces they are moving to Mars next year!",
]

for i, text in enumerate(examples, 1):
    label, probs = predict_news(text)
    print(f"Example {i}: {text}")
    print(f"Prediction: {label}, Probabilities [REAL, FAKE]: {probs}")
    print("-" * 60)

# =========================
# 4️⃣ Interactive input
# =========================
while True:
    text = input("Enter your news text (or 'exit' to quit): ")
    if text.lower() == "exit":
        break
    label, probs = predict_news(text)
    print(f"Prediction: {label}, Probabilities [REAL, FAKE]: {probs}\n")
