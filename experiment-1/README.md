# Generative AI Lab Manual (BAIL657C)

## Experiment 1: Exploring Pre-trained Word Vectors

### Objective:
- Explore pre-trained word vectors using the Sentence Transformer model.
- Perform vector arithmetic and analyze word relationships.
- Evaluate the quality of vector arithmetic using cosine similarity.

### Introduction:
In this experiment, you will explore how words can be represented using numerical vectors. Pre-trained word vectors capture semantic meanings from large datasets. By using vector arithmetic, we can examine how words relate to each other mathematically (e.g., "king - man + woman = queen"). We will also use cosine similarity to measure how close the computed vectors are to actual word embeddings. This helps us analyze the effectiveness of the model in understanding relationships between words.

### Tools and Libraries:
- Python
- PyTorch
- Sentence Transformers (Hugging Face)

### Prerequisites:
Ensure you have Python installed along with the following libraries:

```bash
pip install torch sentence-transformers matplotlib
```

### Step 1: Load Pre-trained Sentence Transformer Model

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example words
words = ['king', 'queen', 'man', 'woman']

# Generate embeddings
embeddings = model.encode(words)

print("Word Embeddings Shape:", embeddings.shape)
```

### Step 2: Perform Vector Arithmetic

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_closest(embedding, embeddings, words):
    similarities = cosine_similarity([embedding], embeddings)[0]
    closest_idx = np.argsort(similarities)[::-1]
    return [(words[i], similarities[i]) for i in closest_idx]

# Vector Arithmetic: king - man + woman
king, man, woman = embeddings[0], embeddings[2], embeddings[3]
new_vector = king - man + woman

# Find closest word
results = find_closest(new_vector, embeddings, words)
print("Closest words to king - man + woman:", results)
```

### Step 3: Visualize Word Embeddings

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 8))
for i, word in enumerate(words):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], word)

plt.title("Word Embeddings Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()
```

### Step 4: Analyze Results
- Observe how similar words cluster together.
- Perform more arithmetic operations and analyze semantic relationships.
- Evaluate the correctness of results using cosine similarity.

### Expected Output
1. Embeddings shape.
2. Closest words based on arithmetic and cosine similarity.
3. PCA visualization plot.

### Conclusion
You have successfully used the Sentence Transformer model to:
- Generate embeddings for words.
- Perform vector arithmetic.
- Evaluate the accuracy using cosine similarity.
- Visualize and interpret the relationships between word embeddings.

This foundation will help in building advanced Generative AI applications.

