# 🧭 HNSW: Hierarchical Navigable Small World

HNSW is a fast and scalable algorithm for **Approximate Nearest Neighbor (ANN)** search, widely used in vector databases to perform semantic search efficiently.

---

## 🔍 What is HNSW?

- A **multi-layer graph** that supports fast similarity search.
- Vectors (nodes) are randomly assigned to one or more levels.
- The higher the level, the fewer nodes — which helps in faster navigation.

---

## 🧠 How It Works (with Diagram)

### 🔼 Node Insertion and Promotion

Nodes are inserted and randomly **promoted** to multiple levels.

Level 3: o
        /
Level 2: o o
         / \ /
Level 1: o---o--o---o
            \ / \ /
Level 0: o--o--o--o--o--o



✅ Nodes at higher levels act like **highways**  
✅ Nodes at level 0 are the **most detailed and dense**

> A node may appear in levels 0, 1, 2 depending on random promotion.

---

## 🔧 Key Concepts

### ✅ Hierarchical Layers
- **Level 0:** All nodes (dense connections)
- **Higher levels:** Fewer nodes (faster traversal)

### ✅ Random Promotion
- Nodes are **randomly promoted** using an exponential distribution.
- This ensures a **logarithmic height** and fast search performance.

### ✅ Navigable Graph
- Each node connects to its **nearest neighbors** in that level.
- Search starts at the top layer and **moves downward** to refine results.

---

## 🧮 Insertion Process

1. Generate a **random level L** for the new node.
2. Insert node into levels: `L, L-1, ..., 0`.
3. At each level, connect it to **k nearest neighbors**.
4. Ensure the graph maintains **short paths and connectivity**.

---

## ⚡ Search Process

1. Begin at **top layer** with a random entry point.
2. Move to the **closest neighbor** in that level.
3. If no closer node is found, move **down one level**.
4. At Level 0, return top K closest vectors.

---

## 📊 Feature Table

| Feature          | HNSW Description                          |
|------------------|--------------------------------------------|
| Structure        | Multi-layer, graph-based                  |
| Node Promotion   | Random (probabilistic)                    |
| Speed            | Fast (log-log scale)                      |
| Accuracy         | High (adjustable)                         |
| Memory Usage     | Higher than brute force (due to links)    |
| Type             | Graph, not a tree                         |

---

## 🔄 Comparison

| Method        | Brute Force         | HNSW                        |
|---------------|---------------------|-----------------------------|
| Time Complexity | O(n)                | ~O(log n)                   |
| Accuracy      | 100%                | 90–99% (approximate)        |
| Speed         | Slow                | Fast                        |
| Use Cases     | Small datasets      | Large-scale vector search   |

---

## ✅ Use Cases

- Semantic Search (PDFs, documents)
- RAG (Retrieval-Augmented Generation)
- Image Search
- Real-time Recommendations

---

## 📚 References

- 📄 Paper: [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- 🔗 Libraries:
  - [FAISS](https://github.com/facebookresearch/faiss)
  - [Weaviate](https://weaviate.io/)
  - [Qdrant](https://qdrant.tech/)
  - [Milvus](https://milvus.io/)
  - [Pinecone](https://www.pinecone.io/)

---
