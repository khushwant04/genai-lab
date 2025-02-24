### **Experiment 2: Visualizing Word Embeddings and Generating Similar Words**

---

### **Objective:**
- Use dimensionality reduction (e.g., PCA or t-SNE) to visualize word embeddings.
- Select 10 words from a specific domain (e.g., sports, technology) and analyze their relationships.
- Generate 5 semantically similar words for a given input using word embeddings.

---

### **Introduction:**
Word embeddings represent words as dense vectors in a multi-dimensional space. These vectors capture the contextual meaning of words based on their usage in large corpora. To interpret these embeddings, dimensionality reduction techniques like PCA (Principal Component Analysis) and t-SNE (t-distributed Stochastic Neighbor Embedding) help project them onto a 2D plane for visualization. 

In this experiment:
1. We visualize word embeddings for a specific domain.
2. We analyze clusters to understand word relationships.
3. We generate semantically similar words using cosine similarity.

---

### **Tools and Libraries:**
- Python
- PyTorch
- Sentence Transformers (Hugging Face)
- scikit-learn (for PCA/t-SNE)
- matplotlib (for visualization)

---

### **Prerequisites:**
Ensure these libraries are installed:

```bash
pip install torch sentence-transformers matplotlib scikit-learn
```

---

### **Step 1: Load Sentence Transformer Model and Define Words**
We use the `sentence-transformers` library to load a pre-trained model and encode domain-specific words.

```python
from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define domain-specific words (technology domain example)
words = [
    "computer", "software", "hardware", "AI", "robotics",
    "internet", "cloud", "cybersecurity", "algorithm", "database"
]

# Generate embeddings
embeddings = model.encode(words)

print("Embeddings shape:", embeddings.shape)  # Output shape (10, 384)
```

---

### **Step 2: Dimensionality Reduction and Visualization**
We use PCA to reduce high-dimensional embeddings into 2D space for visualization.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Visualize embeddings
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], word)

plt.title("PCA Visualization of Technology Domain Words")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()
```

---

### **Step 3: Generate Semantically Similar Words**
We compute cosine similarity to identify the most similar words.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to find the closest words
def find_similar_words(word, model, words_list, top_n=5):
    word_embedding = model.encode([word])
    all_embeddings = model.encode(words_list)
    
    similarities = cosine_similarity(word_embedding, all_embeddings)[0]
    closest_indices = np.argsort(similarities)[::-1][1:top_n + 1]
    
    similar_words = [(words_list[i], similarities[i]) for i in closest_indices]
    return similar_words

# Example: Find 5 similar words to "AI"
similar_words = find_similar_words("AI", model, words)
print("Top 5 words similar to 'AI':", similar_words)
```

---

### **Step 4: Analyze the Output**
- **Visualization Interpretation**: Observe how similar words form clusters (e.g., "AI" and "robotics" should be closer).
- **Similarity Analysis**: Ensure the closest words reflect meaningful relationships in the chosen domain.

---

### **Expected Output:**
1. **Embedding Shape:** `(10, 384)` (10 words with 384-dimensional vectors).
2. **Visualization:** A 2D scatter plot showing how words are grouped based on meaning.
3. **Similar Words:** For the input word "AI", the output may resemble:

   ```
   Top 5 words similar to 'AI': [('robotics', 0.89), ('algorithm', 0.86), ('computer', 0.82), ('software', 0.80), ('cybersecurity', 0.78)]
   ```

---

### **Conclusion:**
In this experiment, you:
1. Encoded domain-specific words using a pre-trained Sentence Transformer model.
2. Applied PCA for 2D visualization to interpret semantic clusters.
3. Generated semantically similar words using cosine similarity.

Understanding how embeddings capture relationships is foundational for advanced Generative AI applications like language models and knowledge retrieval systems.