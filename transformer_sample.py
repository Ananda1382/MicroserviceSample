# Install the transformers library
# !pip install transformers datasets

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 1. Load a Pretrained Model and Tokenizer
model_name = "distilbert-base-uncased"  # DistilBERT model (lightweight version of BERT, can handle many tasks)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification

# 2. Prepare the Dataset
# Load a dataset from Hugging Face (e.g., sentiment classification on IMDB reviews)
dataset = load_dataset("imdb")

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"]).rename_column("label", "labels")

# 3. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 4. Create Trainer and Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()  # This will fine-tune the model on the dataset

# 5. Evaluate the Model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# 6. Make Predictions on New Data
text = "I loved the movie, it was fantastic!"  # Sample input text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print("Predicted Class:", predictions.item())
