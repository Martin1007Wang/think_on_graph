import torch
import pickle
import os
from typing import List, Tuple, Dict
from neo4j import GraphDatabase
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password", 
                 model_name='msmarco-distilbert-base-tas-b'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        try:
            self.driver = GraphDatabase.driver(
                uri, auth=(user, password), max_connection_lifetime=3600,
                max_connection_pool_size=10, connection_acquisition_timeout=60
            )
            self.create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_2_id = None
        self.relation_2_id = None
        self.id_2_entity = None
        self.id_2_relation = None

    def _process_sample(self, sample: dict) -> List[Tuple[str, str, str]]:
        try:
            if not (graph_data := sample.get('graph')):
                return []
            return [triple for triple in graph_data if len(triple) == 3]
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            return []

    def initialize_embeddings(self, dataset: str = "RoG-webqsp", split: str = "train", force_recompute: bool = False):
        logger.info("Initializing embeddings...")
        emb_dir = os.path.join("embeddings", dataset, split, self.model_name)
        os.makedirs(emb_dir, exist_ok=True)

        entity_emb_path = os.path.join(emb_dir, "entity_embeddings.pt")
        entity_2_id_path = os.path.join(emb_dir, "entity_2_id.pkl")
        id_2_entity_path = os.path.join(emb_dir, "id_2_entity.pkl")
        relation_emb_path = os.path.join(emb_dir, "relation_embeddings.pt")
        relation_2_id_path = os.path.join(emb_dir, "relation_2_id.pkl")
        id_2_relation_path = os.path.join(emb_dir, "id_2_relation.pkl")
        
        load_from_disk = (not force_recompute and os.path.exists(entity_emb_path) and 
                         os.path.exists(entity_2_id_path) and os.path.exists(id_2_entity_path) and 
                         os.path.exists(relation_emb_path) and os.path.exists(relation_2_id_path) and 
                         os.path.exists(id_2_relation_path))
        
        load_from_disk = (not force_recompute and os.path.exists(entity_emb_path) and 
                         os.path.exists(entity_2_id_path) and os.path.exists(id_2_entity_path) and 
                         os.path.exists(relation_emb_path) and os.path.exists(relation_2_id_path) and 
                         os.path.exists(id_2_relation_path))
        
        if load_from_disk:
            logger.info("Loading precomputed embeddings from disk...")
            self.entity_embeddings = torch.load(entity_emb_path)
            with open(entity_2_id_path, 'rb') as f:
                self.entity_2_id = pickle.load(f)
            with open(id_2_entity_path, 'rb') as f:
                self.id_2_entity = pickle.load(f)
            self.relation_embeddings = torch.load(relation_emb_path)
            with open(relation_2_id_path, 'rb') as f:
                self.relation_2_id = pickle.load(f)
            with open(id_2_relation_path, 'rb') as f:
                self.id_2_relation = pickle.load(f)
        else:
            with self.driver.session() as session:
                query = "MATCH (n:ENTITY) RETURN DISTINCT n.id AS entity_id"
                result = session.run(query)
                entity_ids = [record["entity_id"] for record in result]
                
                query = "MATCH ()-[r:RELATION]->() RETURN DISTINCT r.type AS relation_type"
                result = session.run(query)
                relation_types = [record["relation_type"] for record in result]
            if entity_ids:
                logger.info(f"Generating embeddings for {len(entity_ids)} entities...")
                self.entity_embeddings = self.model.encode(entity_ids, batch_size=1024, convert_to_tensor=True, show_progress_bar=True)
                self.entity_2_id = {ent: i for i, ent in enumerate(entity_ids)}
                self.id_2_entity = {i: ent for i, ent in enumerate(entity_ids)}
                with open(entity_2_id_path, 'wb') as f:
                    pickle.dump(self.entity_2_id, f)
                with open(id_2_entity_path, 'wb') as f:
                    pickle.dump(self.id_2_entity, f)
            
            if relation_types:
                logger.info(f"Generating embeddings for {len(relation_types)} relations...")
                self.relation_embeddings = self.model.encode(relation_types, batch_size=1024, convert_to_tensor=True, show_progress_bar=True)
                self.relation_2_id = {rel: i for i, rel in enumerate(relation_types)}
                self.id_2_relation = {i: rel for i, rel in enumerate(relation_types)}
                with open(relation_2_id_path, 'wb') as f:
                    pickle.dump(self.relation_2_id, f)
                with open(id_2_relation_path, 'wb') as f:
                    pickle.dump(self.id_2_relation, f)

    def create_indexes(self):
        try:
            with self.driver.session() as session:
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:ENTITY) ON (n.id)")
                session.run("CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)")
            logger.info("Database indexes created.")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")

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

    def load_graph_from_dataset(self, input_file: str, dataset: str = "RoG-webqsp", split: str = "train", batch_size: int = 1024):
        """Load graph data from a dataset and initialize embeddings."""
        logger.info(f"Loading dataset from {input_file}, split: {split}")
        dataset_obj = load_dataset(input_file, split=split)
        logger.info(f"Dataset loaded: {len(dataset_obj)} samples")
        for sample in tqdm(dataset_obj, desc="Processing samples"):
            triples = self._process_sample(sample)
            with self.driver.session() as session:
                self._batch_process(session, triples, batch_size)
        
        with self.driver.session() as session:
            stats = session.run("""
                MATCH (n:ENTITY) WITH count(n) as entity_count
                MATCH ()-[r:RELATION]->() WITH entity_count, count(r) as relation_count
                RETURN entity_count, relation_count
            """).single()
            logger.info(f"Loaded {stats['entity_count']} entities and {stats['relation_count']} relationships")
        
        self.initialize_embeddings(dataset, split)

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
            if direction == "out":
                query = """
                MATCH (source:ENTITY {id: $source_id})-[r:RELATION {type: $relation_type}]->(target:ENTITY)
                RETURN DISTINCT target.id as target_id
                """
            else:  # "in"
                query = """
                MATCH (target:ENTITY)-[r:RELATION {type: $relation_type}]->(source:ENTITY {id: $source_id})
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

    def get_related_relations_by_question(self, entity_id: str, question: str) -> List[Tuple[str, float]]:
        question_emb = self.model.encode(question, convert_to_tensor=True, show_progress_bar=False)
        with self.driver.session() as session:
            query = """
            MATCH (n:ENTITY {id: $entity_id})-[r:RELATION]-()
            RETURN DISTINCT r.type as relation_type
            """
            result = session.run(query, entity_id=entity_id)
            related_relations = {record["relation_type"] for record in result}
        if not related_relations:
            return []
        rel_indices = [self.relation_2_id[rel] for rel in related_relations if rel in self.relation_2_id]
        related_embs = self.relation_embeddings[rel_indices]
        similarities = self.model.similarity(question_emb, related_embs)[0]
        scores, indices = torch.sort(similarities, descending=True)
        result = [(self.id_2_relation[rel_indices[idx.item()]], score.item()) 
                 for score, idx in zip(scores, indices)]
        return result
    
    def get_related_relations(self, entity_id: str) -> List[Tuple[str, float]]:
        with self.driver.session() as session:
            query = """
            MATCH (n:ENTITY {id: $entity_id})-[r:RELATION]-()
            RETURN DISTINCT r.type as relation_type
            """
            result = session.run(query, entity_id=entity_id)
            return [record["relation_type"] for record in result]

    def get_related_entities_by_question(self, question: str, top_k: int = 10) -> List[Tuple[str, float]]:
        question_emb = self.model.encode(question, convert_to_tensor=True, show_progress_bar=False)
        similarities = self.model.similarity(question_emb, self.entity_embeddings)[0]
        scores, indices = torch.topk(similarities, k=min(top_k, len(similarities)), largest=True)
        result = [(self.id_2_entity[idx.item()], score.item())
                for score, idx in zip(scores, indices)]
        return result
