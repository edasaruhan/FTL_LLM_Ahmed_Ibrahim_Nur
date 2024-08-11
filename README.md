
# Automated Assessment of Educational Content Readability Using NLP

## Project Overview

This project aims to support **Sustainable Development Goal 4: Quality Education** by developing an automated system for assessing the readability of educational materials. The primary objective is to ensure that educational content is accessible to a wide audience, including individuals with varying learning abilities. This is achieved by leveraging a pre-trained NLP model to analyze language complexity, sentence structure, and vocabulary, ultimately classifying texts into different readability levels.

## Project Structure

### 1. Research and Setup
- **Model Selection:** We use the `bert-base-uncased` model from Hugging Face, fine-tuned for text classification.
- **Dataset:** A manually created dataset comprising educational texts labeled with corresponding readability levels (easy, medium, difficult).

### 2. Implementation
- **Loading the Model:** The `bert-base-uncased` model is loaded, and the number of labels is set to 3 to classify the text into three readability levels.
- **Data Preparation:** The dataset is split into training and testing sets for model training and evaluation.

### 3. Training the Model
- **Training Arguments:** Defined parameters include batch size, number of epochs, etc. The model is trained using the Hugging Face `Trainer` class.

### 4. Evaluation
- **Metrics:** The model is evaluated using accuracy and F1-Score.
  - **Accuracy:** Represents the ratio of correctly predicted instances to the total number of instances.
  - **F1-Score:** Provides a balance between precision and recall.

### 5. Results
- **Accuracy Calculation:**
  - Correct Predictions: 3
  - Total Predictions: 5
  - Accuracy: 60%
- **F1-Score for Class 0:**
  - Precision: ~0.67
  - Recall: 1
  - F1-Score: ~0.8 (80%)

## Project Title: Fine-tuning a BERT Model for Classifying Medical Research Articles

### Project Overview

This project contributes to **Sustainable Development Goal 3: Good Health and Well-being** by fine-tuning a BERT model to classify medical research articles into categories such as "Prevention," "Treatment," and "Diagnosis." This classification aids healthcare professionals in efficiently reviewing literature by quickly identifying relevant research.

### Project Structure

1. **Setup and Installation:**
   - Install the necessary libraries and tools:
     ```bash
     pip install torch transformers datasets sklearn matplotlib
     ```

2. **Exploratory Data Analysis (EDA):**
   - **Load the Dataset:** Use a dataset of health-related tweets.
   - **Perform Basic EDA:** Analyze sentiment class distribution, check for missing values, and assess tweet lengths.
   - **Visualization:** Use `matplotlib` or `seaborn` to visualize sentiment distribution and tweet lengths.

3. **Dataset Preparation:**
   - **Preprocessing:** Clean text data by removing special characters, converting to lowercase, and tokenizing using the BERT tokenizer.
   - **Label Encoding:** Convert sentiment labels to numerical format (e.g., positive = 0, neutral = 1, negative = 2).

4. **Model Selection:**
   - **Choosing the Model:** Utilize the `bert-base-uncased` model from Hugging Face.
   - **Loading the Model:** Load the BERT model for sequence classification.

5. **Fine-tuning Process:**
   - **Define Training Arguments:** Set hyperparameters like learning rate, number of epochs, and batch size.
   - **Train the Model:** Fine-tune the model using the `Trainer` class from Hugging Face.

6. **Evaluation:**
   - **Test the Model:** Evaluate the model's performance before and after fine-tuning.
   - **Compare Results:** Demonstrate the impact of fine-tuning on task performance.
