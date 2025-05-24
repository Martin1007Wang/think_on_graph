from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import itertools # For combinations

# --- Neo4j Configuration ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Martin1007Wang"

# --- Semantic Similarity Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
# 阈值：用于判断两个关系类型是否“语义相似”到可能导致混淆
CONFUSION_SIMILARITY_THRESHOLD = 0.8 # 这是一个关键参数，需要根据实际情况调整

class Neo4jLLMConfusionFinder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.model = SentenceTransformer(MODEL_NAME)
        print(f"Sentence Transformer model '{MODEL_NAME}' loaded.")

    def close(self):
        self.driver.close()

    def get_all_relationship_types(self):
        print("\nFetching all distinct 'type' property values from 'RELATION' relationships...")
        with self.driver.session() as session:
            # Corrected Cypher query
            result = session.run("""
                MATCH ()-[r:RELATION]-() 
                WHERE r.type IS NOT NULL  // Optional: ensure the property exists
                RETURN DISTINCT r.type AS relationshipType
            """)
            types = [record["relationshipType"] for record in result]
        print(f"Found {len(types)} unique 'type' property values from 'RELATION' relationships.")
        return types

    def get_relationship_embeddings(self, rel_types):
        # (同前一个脚本中的实现)
        if not rel_types:
            return {} # Return empty dict if no rel_types
        print(f"\nGenerating embeddings for {len(rel_types)} relationship types...")
        embeddings = self.model.encode(rel_types, show_progress_bar=True)
        type_to_embedding = {rel_type: embedding for rel_type, embedding in zip(rel_types, embeddings)}
        return type_to_embedding

    def find_confusing_examples(self, type_to_embedding, similarity_threshold):
        if not type_to_embedding:
            print("\nNo relationship type embeddings available.")
            return

        print(f"\nFinding potentially confusing examples for LLMs (similarity threshold: {similarity_threshold})...")
        
        query = """
        MATCH (n:ENTITY)-[r:RELATION]->(m:ENTITY) // Added :ENTITY label for nodes
        WITH n, elementId(n) AS sourceNodeId, properties(n) AS sourceNodeProps,
            collect({
                relType: r.type,
                targetNodeId: elementId(m),
                targetNodeProps: properties(m)
            }) AS outgoingRels
        WHERE size(outgoingRels) >= 2
        RETURN sourceNodeId, sourceNodeProps, outgoingRels
        // LIMIT 5000 // Optional: for large graphs
        """

        confusing_examples_found = []
        nodes_processed_count = 0

        with self.driver.session() as session:
            results = session.run(query)
            for record in results:
                nodes_processed_count += 1
                source_node_id = record["sourceNodeId"]
                source_node_props = dict(record["sourceNodeProps"])
                outgoing_rels = record["outgoingRels"]

                # We need at least two relationships from this node to form a pair for comparison
                if len(outgoing_rels) < 2:
                    continue

                # Iterate over all unique pairs of outgoing relationships from this node
                for rel_a, rel_b in itertools.combinations(outgoing_rels, 2):
                    # Condition: Relationships must point to different target nodes
                    if rel_a["targetNodeId"] == rel_b["targetNodeId"]:
                        continue

                    type_a_str = rel_a["relType"]
                    type_b_str = rel_b["relType"]

                    # Optional: skip if relationship types are identical strings,
                    # unless you want to find cases where the same relationship type
                    # points to different plausible targets for a vague query.
                    # For maximum confusion potential from different types, keep this:
                    if type_a_str == type_b_str:
                        continue

                    emb_a = type_to_embedding.get(type_a_str)
                    emb_b = type_to_embedding.get(type_b_str)

                    if emb_a is not None and emb_b is not None:
                        similarity = cosine_similarity(emb_a.reshape(1, -1), emb_b.reshape(1, -1))[0][0]

                        if similarity >= similarity_threshold:
                            example = {
                                "source_node_id": source_node_id,
                                "source_node_props": source_node_props,
                                "confusing_pair": [
                                    {"type": type_a_str, "target_props": dict(rel_a["targetNodeProps"]), "target_id": rel_a["targetNodeId"]},
                                    {"type": type_b_str, "target_props": dict(rel_b["targetNodeProps"]), "target_id": rel_b["targetNodeId"]}
                                ],
                                "similarity_score": float(similarity) # Convert numpy.float32 to float
                            }
                            confusing_examples_found.append(example)
                if nodes_processed_count % 100 == 0:
                    print(f"  Processed {nodes_processed_count} source nodes...")
        
        print(f"Finished processing. Found {len(confusing_examples_found)} potentially confusing examples.")

        if not confusing_examples_found:
            print("No confusing examples found with the current threshold.")
            return
            
        print("\n--- Potentially Confusing Examples for LLMs ---")
        # Sort by similarity score for better presentation (most confusing first)
        confusing_examples_found.sort(key=lambda x: x["similarity_score"], reverse=True)

        for i, ex in enumerate(confusing_examples_found[:20]): # Print top 20 examples
            print(f"\nExample {i+1} (Similarity: {ex['similarity_score']:.4f}):")
            print(f"  Source Node (ID: {ex['source_node_id']}): {ex['source_node_props']}")
            
            rel1_info = ex['confusing_pair'][0]
            rel2_info = ex['confusing_pair'][1]
            
            print(f"    -[ {rel1_info['type']} ]-> Target (ID: {rel1_info['target_id']}): {rel1_info['target_props']}")
            print(f"    -[ {rel2_info['type']} ]-> Target (ID: {rel2_info['target_id']}): {rel2_info['target_props']}")
            print(f"  Reasoning: Relationship types '{rel1_info['type']}' and '{rel2_info['type']}' are semantically similar.")
        print("--- End of Examples ---")
        if len(confusing_examples_found) > 20:
            print(f"... and {len(confusing_examples_found) - 20} more examples.")


def main():
    finder = None
    try:
        finder = Neo4jLLMConfusionFinder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        all_rel_types = finder.get_all_relationship_types()
        if not all_rel_types:
            print("No relationship types found. Exiting.")
            return

        type_to_embedding = finder.get_relationship_embeddings(all_rel_types)
        if not type_to_embedding:
            print("Could not generate embeddings. Exiting.")
            return
        
        finder.find_confusing_examples(type_to_embedding, CONFUSION_SIMILARITY_THRESHOLD)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if finder:
            finder.close()
            print("\nNeo4j connection closed.")

if __name__ == "__main__":
    main()