import faiss
import numpy as np

# Example datasets (You can load your own datasets)
corpus = ["This is a sample sentence.", "Another example of a sentence.",
          "This is yet another sentence.", "More sentences for testing."]

# Function to get BERT embeddings (reused from Part 1)
from transformers import BertTokenizer, BertModel

def get_bert_embeddings(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())
    return np.array(embeddings)

# Get embeddings for the corpus
corpus_embeddings = get_bert_embeddings(corpus)

# Build the FAISS index
d = corpus_embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)
index.add(corpus_embeddings)

def search(query, top_k=3):
    query_embedding = get_bert_embeddings([query])[0].reshape(1, -1)
    D, I = index.search(query_embedding, top_k)
    return [(corpus[i], D[0][i]) for i in I[0]]

def main():
    query = input("Enter your search query: ")
    results = search(query)
    print("\nSearch Results:")
    for result in results:
        print(f"Document: {result[0]}, Score: {result[1]}")

if __name__ == "__main__":
    main()
