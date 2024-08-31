from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Example datasets (You can load your own datasets)
dataset_1 = ["This is a sample sentence.", "Another example of a sentence."]
dataset_2 = ["This is yet another sentence.", "More sentences for testing."]

# Function to get Word2Vec embeddings
def get_word2vec_embeddings(sentences):
    model = Word2Vec(sentences=[s.split() for s in sentences], vector_size=100, min_count=1)
    return model

# Function to get GloVe embeddings (You may need to download a pre-trained GloVe model)
def get_glove_embeddings():
    glove_vectors = KeyedVectors.load_word2vec_format('glove.6B.100d.txt', binary=False, no_header=True)
    return glove_vectors

# Function to get BERT embeddings
def get_bert_embeddings(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
    return np.array(embeddings)

# Function to compare embeddings (example using cosine similarity)
def compare_embeddings(embedding_1, embedding_2):
    similarity = np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
    return similarity

def main():
    # Get embeddings for each dataset
    word2vec_model = get_word2vec_embeddings(dataset_1)
    glove_model = get_glove_embeddings()
    bert_embeddings = get_bert_embeddings(dataset_1)
    
    # Compare embeddings (Example with BERT)
    for sentence_1, sentence_2 in zip(dataset_1, dataset_2):
        emb1 = get_bert_embeddings([sentence_1])[0]
        emb2 = get_bert_embeddings([sentence_2])[0]
        print(f"Similarity between '{sentence_1}' and '{sentence_2}': {compare_embeddings(emb1, emb2)}")
        
if __name__ == "__main__":
    main()
