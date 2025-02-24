# **Generative AI - Semester 6 (BAIL657C)**

## **Lab Manual: Experiment 1**

**Title:** Exploring Pre-Trained Word Vectors and Analyzing Word Relationships using Vector Arithmetic

**Objective:**
- Understand and explore pre-trained word vectors.
- Perform vector arithmetic to analyze word relationships.
- Interpret results from arithmetic operations on word embeddings.

**Prerequisites:**
- Basic understanding of word embeddings.
- Python programming proficiency.
- Familiarity with PyTorch and Hugging Face libraries.

**Tools & Libraries Required:**
- Python (>=3.10)
- PyTorch (>=2.0)
- Hugging Face Transformers (latest version)
- Matplotlib (for visualization)
- Google Colab or local Python environment (optional: Jupyter Notebook)

---

## **Theory:**
Word embeddings are dense vector representations of words. Pre-trained word vectors capture semantic relationships between words. By performing vector arithmetic on these embeddings, we can reveal analogies and relationships (e.g., "king" - "man" + "woman" = "queen").

Common pre-trained models include:
- GloVe (Global Vectors for Word Representation)
- Word2Vec (Google's model for learning word embeddings)
- FastText (Facebook's extension of Word2Vec)

---

## **Experiment Procedure:**

### **Step 1: Environment Setup**
Ensure Python and required libraries are installed.

```bash
# Install required packages
pip install torch transformers matplotlib
```

### **Step 2: Load Pre-Trained Word Vectors**
We'll use the `bert-base-uncased` model from Hugging Face for word embeddings.

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

print("Model and tokenizer loaded successfully")
```

### **Step 3: Extract Word Embeddings**

```python
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    # Extract the embedding of the [CLS] token
    return outputs.last_hidden_state[:, 0, :].detach()

# Example words
word1 = "king"
word2 = "man"
word3 = "woman"

emb_king = get_word_embedding(word1)
emb_man = get_word_embedding(word2)
emb_woman = get_word_embedding(word3)

print("Word embeddings extracted")
```

### **Step 4: Perform Vector Arithmetic**

```python
# king - man + woman = ?
result_vector = emb_king - emb_man + emb_woman

print("Vector arithmetic performed")
```

### **Step 5: Find the Closest Word**

```python
import numpy as np

def find_closest_word(target_embedding, words):
    closest_word = None
    min_distance = float('inf')

    for word in words:
        emb = get_word_embedding(word)
        distance = torch.norm(target_embedding - emb)

        if distance < min_distance:
            min_distance = distance
            closest_word = word

    return closest_word

# Candidate words to search
word_list = ["queen", "prince", "princess", "king", "man", "woman"]

closest_word = find_closest_word(result_vector, word_list)
print(f"Closest word to 'king - man + woman' is: {closest_word}")
```

### **Step 6: Visualize Word Relationships (Optional)**

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Collect embeddings and words
words = ["king", "man", "woman", "queen"]
embeddings = torch.cat([get_word_embedding(word) for word in words])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings.numpy())

# Plot embeddings
plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], word, fontsize=12)

plt.title("Word Embedding Visualization")
plt.grid(True)
plt.show()
```

---

## **Expected Output:**
1. Successfully loaded pre-trained word vectors.
2. Perform vector arithmetic and identify the closest semantic word.
3. Visualize embeddings in a 2D space using PCA.

**Example Output:**
```
Model and tokenizer loaded successfully
Word embeddings extracted
Vector arithmetic performed
Closest word to 'king - man + woman' is: queen
```

---

## **Result Analysis:**
- The model correctly identifies the analogy "king - man + woman = queen".
- Visualization helps in understanding word clusters and semantic relationships.

**Key Insights:**
- Word embeddings capture semantic similarity.
- Vector arithmetic on embeddings can reveal meaningful relationships.
- Dimensionality reduction aids in visualizing high-dimensional data.

---

## **Conclusion:**
In this experiment, we successfully explored pre-trained word vectors, performed vector arithmetic, and analyzed the results. This knowledge forms the foundation for advanced generative AI tasks, including language modeling and prompt engineering.

**Further Exploration:**
- Train a custom Word2Vec model using domain-specific datasets.
- Explore contextual embeddings with models like GPT or BERT.
- Apply dimensionality reduction using t-SNE for better visualization.

---

**References:**
1. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
2. Mikolov, T., et al. "Efficient Estimation of Word Representations in Vector Space." arXiv preprint 2013.
3. Hugging Face Documentation: https://huggingface.co/docs
4. PyTorch Documentation: https://pytorch.org/docs

---