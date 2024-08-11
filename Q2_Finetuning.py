import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from datasets import load_metric
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Install Necessary Libraries and Tools
# Already done: torch, transformers, datasets, sklearn, matplotlib

# 2. Load the Dataset and Perform Basic EDA
# Assuming a dataset of health-related articles with labels for "Prevention", "Treatment", and "Diagnosis"
dataset = load_dataset("health_fact")

# Simplify the dataset for demonstration purposes
df = dataset['train'].to_pandas()
df = df[['text', 'label']]

# Perform Basic EDA
# Analyze the distribution of sentiment classes
print(df['label'].value_counts())

# Check for missing values
print(df.isnull().sum())

# Understand the length of text articles
df['text_length'] = df['text'].apply(len)
print(df['text_length'].describe())

# Visualization: Distribution of sentiment classes and text lengths
sns.countplot(x=df['label'])
plt.title("Distribution of Article Categories")
plt.show()

sns.histplot(df['text_length'], bins=20, kde=True)
plt.title("Distribution of Article Lengths")
plt.show()

# 3. Dataset Preparation
# Preprocessing: Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128)

df['label'] = df['label'].map({"Prevention": 0, "Treatment": 1, "Diagnosis": 2})
dataset = DatasetDict({
    'train': df[['text', 'label']]
})

tokenized_datasets = dataset.map(lambda x: tokenize_function(x['text']), batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# Train-test split
train_dataset, test_dataset = train_test_split(tokenized_datasets['train'], test_size=0.2, random_state=42)

# Convert to Dataset objects
train_dataset = DatasetDict({'train': train_dataset})
test_dataset = DatasetDict({'test': test_dataset})

# 4. Model Selection
# Choosing and Loading the Pre-trained Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)

# 5. Fine-tuning Process
# Define Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset['train'],
    eval_dataset=test_dataset['test'],
    compute_metrics=lambda p: load_metric("accuracy").compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
)

trainer.train()

# 6. Evaluation
# Test the Model and Compare Before and After Fine-tuning
eval_results_before = trainer.evaluate(eval_dataset=test_dataset['test'])
print(f"Test Accuracy Before Fine-tuning: {eval_results_before['eval_accuracy']}")

# Fine-tune the model
trainer.train()

# Evaluate the fine-tuned model
eval_results_after = trainer.evaluate(eval_dataset=test_dataset['test'])
print(f"Test Accuracy After Fine-tuning: {eval_results_after['eval_accuracy']}")

# Comparison
improvement = eval_results_after['eval_accuracy'] - eval_results_before['eval_accuracy']
print(f"Accuracy Improvement After Fine-tuning: {improvement}")