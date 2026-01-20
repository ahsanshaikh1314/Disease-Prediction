import os
from neo4j import GraphDatabase
# CHANGE 1: Import DiscreteBayesianNetwork instead of BayesianNetwork
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# --- Task 1: Setup and Configuration ---
# NOTE: Update these with your Neo4j database credentials
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678" 

KNOWLEDGE_FILE = "knowledge.txt"

class MedicalKnowledgeGraph:
    """
    Manages connection and operations with the Neo4j graph database.
    Corresponds to Tasks 1, 2, 5, 7.
    """
    def __init__(self, uri, user, password):
        # Initialize the Neo4j driver
        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            print("Successfully connected to Neo4j.")
        except Exception as e:
            print(f"Failed to create Neo4j driver: {e}")
            self._driver = None

    def close(self):
        # Close the driver connection
        if self._driver is not None:
            self._driver.close()
            print("Neo4j connection closed.")

    def _execute_query(self, query, parameters=None):
        # Helper function to execute a Cypher query
        if self._driver is None:
            print("Driver not initialized.")
            return None
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def clear_database(self):
        # Clears all nodes and relationships from the database
        print("Clearing the database...")
        query = "MATCH (n) DETACH DELETE n"
        self._execute_query(query)
        print("Database cleared.")

    def add_knowledge(self, disease, symptom):
        # Task 5: Generate and execute Neo4j queries to add nodes and relationships
        # Using MERGE to avoid creating duplicate nodes
        query = (
            "MERGE (d:Disease {name: $disease_name}) "
            "MERGE (s:Symptom {name: $symptom_name}) "
            "MERGE (d)-[:HAS_SYMPTOM]->(s)"
        )
        self._execute_query(query, parameters={'disease_name': disease, 'symptom_name': symptom})

    def query_diseases_by_symptoms(self, symptoms):
        # Task 7: Retrieve information from Neo4j
        if not symptoms:
            return []
        # This query finds diseases that have ALL of the specified symptoms
        query = """
        MATCH (d:Disease)
        WHERE ALL(symptom_name IN $symptoms WHERE EXISTS((d)-[:HAS_SYMPTOM]->(:Symptom {name: symptom_name})))
        RETURN d.name AS disease
        """
        results = self._execute_query(query, parameters={'symptoms': symptoms})
        return [record['disease'] for record in results]


class KnowledgeParser:
    """
    Handles reading and parsing the knowledge from a text file.
    Corresponds to Tasks 3 and 4.
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def parse_knowledge(self):
        """
        Reads the knowledge file and parses it into a dictionary.
        Returns:
            dict: A dictionary mapping diseases to a list of their symptoms.
        """
        knowledge = {}
        print(f"\n--- Task 3: Reading Text File '{self.filepath}' ---")
        try:
            with open(self.filepath, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: The file '{self.filepath}' was not found.")
            return {}

        print("\n--- Task 4: Parsing Sentences ---")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if " has symptoms " in line:
                parts = line.split(" has symptoms ")
                disease = parts[0].strip()
                symptoms_str = parts[1].replace('.', '').strip()
                symptoms = [s.strip() for s in symptoms_str.split(',')]
                
                if disease not in knowledge:
                    knowledge[disease] = []
                knowledge[disease].extend(symptoms)
                print(f"Parsed: Disease='{disease}', Symptoms={symptoms}")
        
        return knowledge


class BayesianDiagnosisModel:
    """
    Builds and queries the Bayesian Network for medical diagnosis.
    Corresponds to Task 6.
    """
    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.diseases = list(knowledge.keys())
        self.symptoms = sorted(list(set(s for sym_list in knowledge.values() for s in sym_list)))
        self.model = self._build_model()
        self.inference = VariableElimination(self.model)

    def _build_model(self):
        print("\n--- Task 6: Building the Bayesian Network ---")
        
        model_structure = []
        for disease, symptoms in self.knowledge.items():
            for symptom in symptoms:
                model_structure.append((disease, symptom))
        
        # CHANGE 2: Use DiscreteBayesianNetwork instead of BayesianNetwork
        model = DiscreteBayesianNetwork(model_structure)
        print("Model Structure (Edges):", model.edges())
        
        # --- Define Conditional Probability Distributions (CPDs) ---
        
        for disease in self.diseases:
            cpd = TabularCPD(variable=disease, variable_card=2, values=[[0.99], [0.01]])
            model.add_cpds(cpd)

        for symptom in self.symptoms:
            parents = sorted(model.get_parents(symptom))
            cardinality = [2] * len(parents)
            
            prob_symptom_true = []
            num_parent_states = 2 ** len(parents)
            
            for i in range(num_parent_states):
                parent_states = [int(x) for x in bin(i)[2:].zfill(len(parents))]
                prob_symptom_false_given_parents = 1.0
                leak_prob = 0.01
                prob_inhibition = 0.2 

                for j, state in enumerate(parent_states):
                    if state == 1:
                        prob_symptom_false_given_parents *= prob_inhibition
                
                final_prob_true = 1.0 - (prob_symptom_false_given_parents * (1 - leak_prob))
                prob_symptom_true.append(final_prob_true)

            values = [
                [1 - p for p in prob_symptom_true],
                prob_symptom_true
            ]
            
            cpd = TabularCPD(variable=symptom, variable_card=2,
                             values=values,
                             evidence=parents, evidence_card=cardinality)
            model.add_cpds(cpd)

        print("Checking if model is valid...")
        if model.check_model():
            print("Bayesian Network model is valid.")
        else:
            print("Error: Bayesian Network model is not valid.")
            
        return model

    def diagnose(self, observed_symptoms):
        print(f"\n--- Performing Diagnosis for Observed Symptoms: {observed_symptoms} ---")
        
        evidence = {symptom: 1 for symptom in observed_symptoms}
        
        # Filter out symptoms not in the model from evidence
        evidence = {k: v for k, v in evidence.items() if k in self.model.nodes()}

        query_diseases = [d for d in self.diseases if d in self.model.nodes()]
        
        try:
            # Querying each disease probability individually for clearer output
            print("Posterior Probabilities of Diseases:")
            for disease in query_diseases:
                result = self.inference.query(variables=[disease], evidence=evidence)
                prob_str = f"  - P({disease}=True | evidence): {result.values[1]:.4f}"
                print(prob_str)

        except Exception as e:
            print(f"Error during inference: {e}")


def main():
    """Main function to run the entire project pipeline."""
    
    mkg = MedicalKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    if mkg._driver is None:
        return 
    
    mkg.clear_database()

    parser = KnowledgeParser(KNOWLEDGE_FILE)
    knowledge_dict = parser.parse_knowledge()
    if not knowledge_dict:
        mkg.close()
        return

    print("\n--- Task 5: Populating Neo4j Knowledge Graph ---")
    for disease, symptoms in knowledge_dict.items():
        for symptom in symptoms:
            mkg.add_knowledge(disease, symptom)
    print("Neo4j graph has been populated.")

    print("\n--- Task 7: Querying the Knowledge Graph ---")
    test_symptoms = ['Fever', 'Cough']
    print(f"Querying for diseases that have ALL symptoms: {test_symptoms}")
    diseases = mkg.query_diseases_by_symptoms(test_symptoms)
    if diseases:
        print("Found matching diseases in Neo4j:", diseases)
    else:
        print("No diseases found in Neo4j with all specified symptoms.")
    
    bn_model = BayesianDiagnosisModel(knowledge_dict)
    
    bn_model.diagnose(observed_symptoms=['Headache'])
    # bn_model.diagnose(observed_symptoms=['Sore Throat', 'Fever'])
    # bn_model.diagnose(observed_symptoms=['Sneezing', 'Runny Nose'])

   
    mkg.close()

if __name__ == "__main__":
    main()