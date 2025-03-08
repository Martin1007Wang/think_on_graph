import torch
import json
import os
from typing import List, Tuple, Dict
from neo4j import GraphDatabase
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def process_sample(sample: dict) -> List[Tuple[str, str, str]]:
    try:
        if not (graph_data := sample.get('graph')):
            return []
        return [triple for triple in graph_data if len(triple) == 3]
    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        return []

class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password", 
                 dataset="RoG-webqsp", split="train", model_name='all-MiniLM-L6-v2'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        try:
            self.driver = GraphDatabase.driver(
                uri, auth=(user, password), max_connection_lifetime=3600,
                max_connection_pool_size=10, connection_acquisition_timeout=60
            )
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        self.entity_embeddings = None
        self.entity_ids = None
        self.relation_embeddings = None
        self.relation_types = None
        self._initialize_embeddings(dataset, split)

    def _initialize_embeddings(self, dataset: str = "RoG-webqsp", split: str = "train"):
        logger.info("Initializing embeddings...")
        emb_dir = os.path.join("embeddings", dataset, split, self.model_name)
        os.makedirs(emb_dir, exist_ok=True)

        entity_emb_path = os.path.join(emb_dir, "entity_embeddings.npy")
        entity_ids_path = os.path.join(emb_dir, "entity_ids.json")
        relation_emb_path = os.path.join(emb_dir, "relation_embeddings.npy")
        relation_types_path = os.path.join(emb_dir, "relation_types.json")
        
        load_from_disk = (os.path.exists(entity_emb_path) and os.path.exists(entity_ids_path) and
                          os.path.exists(relation_emb_path) and os.path.exists(relation_types_path))
        
        if load_from_disk:
            logger.info("Loading precomputed embeddings from disk...")
            self.entity_embeddings = np.load(entity_emb_path)
            with open(entity_ids_path, 'r') as f:
                self.entity_ids = json.load(f)
            self.relation_embeddings = np.load(relation_emb_path)
            with open(relation_types_path, 'r') as f:
                self.relation_types = json.load(f)
        else:
            with self.driver.session() as session:
                query = "MATCH (n:ENTITY) RETURN DISTINCT n.id AS entity_id"
                result = session.run(query)
                self.entity_ids = [record["entity_id"] for record in result]
                
                query = "MATCH ()-[r:RELATION]->() RETURN DISTINCT r.type AS relation_type"
                result = session.run(query)
                self.relation_types = [record["relation_type"] for record in result]
            
            if not self.entity_ids:
                logger.warning("No entities found in the database.")
            if not self.relation_types:
                logger.warning("No relations found in the database.")

            if self.entity_ids:
                logger.info(f"Generating embeddings for {len(self.entity_ids)} entities...")
                self.entity_embeddings = self.model.encode(
                    self.entity_ids, batch_size=256, convert_to_numpy=True, show_progress_bar=True
                ).astype(np.float32)
                norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
                self.entity_embeddings = self.entity_embeddings / np.where(norms > 0, norms, 1)  # 避免除以零
                np.save(entity_emb_path, self.entity_embeddings)
                with open(entity_ids_path, 'w') as f:
                    json.dump(self.entity_ids, f)
                logger.info(f"Entity embeddings saved to {entity_emb_path}")
            
            if self.relation_types:
                logger.info(f"Generating embeddings for {len(self.relation_types)} relations...")
                processed_relations = [rel.replace('.', ' ') for rel in self.relation_types]
                self.relation_embeddings = self.model.encode(
                    processed_relations, batch_size=256, convert_to_numpy=True, show_progress_bar=True
                ).astype(np.float32)
                # 预归一化
                norms = np.linalg.norm(self.relation_embeddings, axis=1, keepdims=True)
                self.relation_embeddings = self.relation_embeddings / np.where(norms > 0, norms, 1)
                np.save(relation_emb_path, self.relation_embeddings)
                with open(relation_types_path, 'w') as f:
                    json.dump(self.relation_types, f)
                logger.info(f"Relation embeddings saved to {relation_emb_path}")

    def create_indexes(self):
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:ENTITY) ON (n.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)")

    def close(self):
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            logger.info("Clearing database...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared.")

    def _batch_process(self, session, triples, batch_size: int = 500):
        def _run_batch(tx):
            query = """
            UNWIND $batch as triple
            MERGE (head:ENTITY {id: triple[0]})
            MERGE (tail:ENTITY {id: triple[2]})
            MERGE (head)-[r:RELATION {type: triple[1]}]->(tail)
            """
            tx.run(query, batch=batch)

        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            if batch:
                session.execute_write(_run_batch)

    def load_graph_from_dataset(self, input_file: str, split: str = "train", batch_size: int = 1024):
        self.create_indexes()
        logger.info(f"Loading dataset from {input_file}, split: {split}")
        dataset = load_dataset(input_file, split=split)
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        for sample in tqdm(dataset, desc="Processing samples"):
            triples = process_sample(sample)
            with self.driver.session() as session:
                self._batch_process(session, triples, batch_size)
        with self.driver.session() as session:
            stats = session.run("""
                MATCH (n:ENTITY) WITH count(n) as entity_count
                MATCH ()-[r:RELATION]->() WITH entity_count, count(r) as relation_count
                RETURN entity_count, relation_count
            """).single()
            logger.info(f"Loaded {stats['entity_count']} entities and {stats['relation_count']} relationships")
        dataset_name = os.path.basename(input_file)
        self._initialize_embeddings(dataset_name, split)

    def get_shortest_paths(self, source_id: str, target_id: str, max_depth: int = 5) -> List[List[Dict]]:
        with self.driver.session() as session:
            if source_id == target_id:
                query = """
                MATCH (source:ENTITY {id: $source_id})-[r]->(source)
                RETURN collect({source: $source_id, relation: r.type, target: $source_id}) as loops
                """
                result = session.run(query, source_id=source_id)
                loops = result.single()["loops"]
                return [loops] if loops else []
            
            query = """
            MATCH (source:ENTITY {id: $source_id}), (target:ENTITY {id: $target_id})
            MATCH paths = shortestPath((source)-[*]->(target))
            RETURN paths
            """
            paths = []
            results = session.run(query, source_id=source_id, target_id=target_id, max_depth=max_depth)
            for record in results:
                path = record["paths"]
                path_info = []
                nodes = path.nodes
                rels = path.relationships
                for i, (node, rel) in enumerate(zip(nodes[:-1], rels)):
                    path_info.append({
                        'source': node['id'],
                        'relation': rel['type'],
                        'target': nodes[i + 1]['id']
                    })
                paths.append(path_info)
            return paths

    def get_target_entities(self, source_id: str, relation_type: str, direction: str = "out") -> List[str]:
        with self.driver.session() as session:
            query = """
            MATCH (source:ENTITY {id: $source_id})-[r:RELATION {type: $relation_type}]->(target:ENTITY)
            RETURN DISTINCT target.id as target_id
            """
            result = session.run(query, source_id=source_id, relation_type=relation_type)
            return [record["target_id"] for record in result]

    def get_all_entities(self):
        with self.driver.session() as session:
            query = "MATCH (n:ENTITY) RETURN DISTINCT n.id AS entity"
            result = session.run(query)
            return [record["entity"] for record in result]

    def get_all_relations(self):
        with self.driver.session() as session:
            query = "MATCH ()-[r:RELATION]->() RETURN DISTINCT r.type AS relation"
            result = session.run(query)
            return [record["relation"] for record in result]

    def get_related_relations_by_question(self, entity_id: str, question: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.relation_embeddings is None:
            logger.warning("Relation embeddings not initialized.")
            return []
        
        question_emb = self.model.encode(
            question, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        question_norm = np.linalg.norm(question_emb)
        if question_norm == 0:
            logger.warning("Question embedding norm is zero.")
            return []
        question_emb = question_emb / question_norm
        
        with self.driver.session() as session:
            query = """
            MATCH (n:ENTITY {id: $entity_id})-[r:RELATION]-()
            RETURN DISTINCT r.type as relation_type
            """
            related_relations = [record["relation_type"] for record in session.run(query, entity_id=entity_id)]
        
        if not related_relations:
            return []
        
        rel_indices = [self.relation_types.index(rel) for rel in related_relations if rel in self.relation_types]
        if not rel_indices:
            return []
        
        related_embs = self.relation_embeddings[rel_indices]
        sims = np.dot(related_embs, question_emb)

        top_indices = np.argpartition(-sims, top_k-1)[:top_k]
        top_sims = sims[top_indices]
        top_relations = [related_relations[i] for i in top_indices]
        similarities = [(rel, float(sim)) for rel, sim in zip(top_relations, top_sims)]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def get_similar_entities(self, entity_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.entity_embeddings is None or entity_id not in self.entity_ids:
            logger.warning(f"Entity embeddings not initialized or {entity_id} not found.")
            return []
        
        query_emb = self.model.encode(
            [entity_id], convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        query_emb = query_emb.flatten()
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            logger.warning("Query embedding norm is zero.")
            return []
        query_emb = query_emb / query_norm
        
        sims = np.dot(self.entity_embeddings, query_emb)
        top_indices = np.argpartition(-sims, top_k)[:top_k+1]
        top_sims = sims[top_indices]
        top_entities = [self.entity_ids[i] for i in top_indices]
        
        similarities = []
        for entity, sim in zip(top_entities, top_sims):
            if entity != entity_id:
                similarities.append((entity, float(sim)))
            if len(similarities) >= top_k:
                break
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def get_similar_entities_by_question(self, entity_id: str, question: str, n: int = 5) -> List[Tuple[str, float]]:
        if self.entity_embeddings is None:
            logger.warning("Entity embeddings not initialized.")
            return []
        
        question_emb = self.model.encode(
            question, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        question_norm = np.linalg.norm(question_emb)
        if question_norm == 0:
            logger.warning("Question embedding norm is zero.")
            return []
        question_emb = question_emb / question_norm
        
        sims = np.dot(self.entity_embeddings, question_emb)
        top_indices = np.argpartition(-sims, n-1)[:n]
        top_sims = sims[top_indices]
        top_entities = [self.entity_ids[i] for i in top_indices]
        
        similarities = [(entity, float(sim)) for entity, sim in zip(top_entities, top_sims)]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='rmanluo')
    parser.add_argument('--dataset', '-d', type=str, default='RoG-webqsp')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687')
    parser.add_argument('--neo4j_user', type=str, default='neo4j')
    parser.add_argument('--neo4j_password', type=str, default='Martin1007Wang')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='Pretrained model name')
    parser.add_argument('--clear', type=bool, default=True, help='Clear database before loading')
    
    args = parser.parse_args()

    logger.info("Running with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    kg = KnowledgeGraph(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        model_name=args.model_name,
        dataset=args.dataset,
        split=args.split
    )
    
    try:
        if args.clear:
            kg.clear_database()

        input_file = os.path.join(args.data_path, args.dataset)
        logger.info(f"Full dataset path: {input_file}")

        if not os.path.exists(input_file):
            logger.warning(f"Warning: {input_file} does not exist as a file")

        kg.load_graph_from_dataset(input_file, args.split)
    finally:
        kg.close()

if __name__ == "__main__":
    main()