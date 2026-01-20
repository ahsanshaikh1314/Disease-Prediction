# Hybrid Disease Prediction System (Neo4j + Bayesian Networks)

A hybrid AI diagnostic engine that combines **Graph Databases (Neo4j)** for structured knowledge representation with **Probabilistic Graphical Models (pgmpy)** for reasoning under uncertainty.

## 🚀 Overview
This project solves a common problem in medical expert systems: how to handle complex, multi-symptom diseases where symptoms are not deterministic.

Instead of simple "If-Then" rules, this system:
1.  **Parses Unstructured Rules:** Reads medical knowledge from plain text (e.g., "Flu has symptoms Fever, Cough").
2.  **Builds a Knowledge Graph:** visualizes relationships in **Neo4j** (Disease -> Symptom).
3.  **Performs Probabilistic Inference:** dynamically constructs a **Bayesian Network** to calculate the probability of a disease given partial evidence (e.g., "Patient has Headache").

## 🧠 Key Technical Features
* **Graph-Based Storage:** Utilizes Neo4j to maintain a flexible schema of Diseases and Symptoms.
* **Noisy-OR Gate Logic:** Implements custom Conditional Probability Distributions (CPDs) using "Noisy-OR" logic. This models real-world medical uncertainty (e.g., a disease usually causes a symptom, but not always).
    * *Inhibition Probability:* 0.2
    * *Leak Probability:* 0.01 (Spontaneous symptom occurrence)
* **Exact Inference:** Uses **Variable Elimination** algorithms to compute exact posterior probabilities $P(Disease | Evidence)$.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Database:** Neo4j (Graph Database)
* **AI/ML Libraries:**
    * `pgmpy` (Probabilistic Graphical Models)
    * `neo4j` (Python Driver)

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/disease-prediction-system.git](https://github.com/yourusername/disease-prediction-system.git)
    cd disease-prediction-system
    ```

2.  **Install dependencies**
    ```bash
    pip install neo4j pgmpy
    ```

3.  **Setup Neo4j**
    * Install Neo4j Desktop or use a Sandbox instance.
    * Start the database on `bolt://localhost:7687`.
    * Update the credentials in `medical_diagnosis_project.py`:
        ```python
        NEO4J_URI = "neo4j://127.0.0.1:7687"
        NEO4J_USER = "neo4j"
        NEO4J_PASSWORD = "your_password"
        ```

## 📝 Usage

1.  **Prepare the Knowledge Base**
    Create a `knowledge.txt` file in the root directory with rules:
    ```text
    Flu has symptoms Fever, Cough, Fatigue
    Cold has symptoms Cough, Sneezing, Runny Nose
    Covid has symptoms Fever, Cough, Loss of Taste
    ```

2.  **Run the Pipeline**
    ```bash
    python medical_diagnosis_project.py
    ```

3.  **Output Example**
    ```text
    --- Performing Diagnosis for Observed Symptoms: ['Fever'] ---
    Posterior Probabilities:
      - P(Flu=True | Fever): 0.4521
      - P(Covid=True | Fever): 0.4103
      - P(Cold=True | Fever): 0.0012
    ```

## 📂 Project Structure
* `medical_diagnosis_project.py`: Main driver code containing the `MedicalKnowledgeGraph`, `KnowledgeParser`,
