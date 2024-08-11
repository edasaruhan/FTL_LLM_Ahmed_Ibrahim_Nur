import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import load_metric

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)

# Create or Load the Dataset (for example purposes, let's create a small sample dataset)
texts = [
    "This is an easy sentence to understand.",
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms.",
    "The cat sat on the mat."
]
labels = [0, 2, 0]  # 0 = easy, 1 = medium, 2 = difficult

# Tokenize the data
def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', truncation=True)

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

# Convert the tokenized data to a Dataset
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# Define a function for computing metrics (accuracy in this case)
def compute_metrics(eval_pred):
from datasets import load_metric

# Simulate some predictions and true labels
predictions = np.array([0, 2, 0, 0, 1])
true_labels = np.array([0, 2, 0, 1, 2])

# Load the metric
accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

# Compute the metrics
accuracy = accuracy_metric.compute(predictions=predictions, references=true_labels)
f1 = f1_metric.compute(predictions=predictions, references=true_labels, average="macro")

print(f"Accuracy: {accuracy['accuracy']:.2f}")
print(f"F1-Score: {f1['f1']:.2f}")
# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the test dataset
eval_result = trainer.evaluate()

print(f"Test Accuracy: {eval_result['eval_accuracy']}")
