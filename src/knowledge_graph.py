import torch
import pickle # Still needed if you were to save/load mappings manually, but not for this version
import os
import json
from typing import List, Tuple, Dict, Optional, Any
from neo4j import GraphDatabase # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from tqdm import tqdm # type: ignore
import logging

# Attempt to import FAISS
try:
    import faiss # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING) # type: ignore
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.model: Optional[SentenceTransformer] = None
        self.model_name: Optional[str] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"KnowledgeGraph will use device: {self.device} for SentenceTransformer models when loaded.")
        if FAISS_AVAILABLE:
            logger.info("FAISS library is available and will be used for efficient entity search.")
        else:
            logger.warning("FAISS library is not installed. Semantic entity search ('get_related_entities_by_question') will be unavailable or fallback to slow exact search if implemented.")

        try:
            self.driver = GraphDatabase.driver(
                uri, auth=(user, password), max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
                keep_alive=True
            )
            self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {uri}")
            self.create_indexes()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j or create indexes: {e}", exc_info=True)
            raise

        self.entity_embeddings: Optional[torch.Tensor] = None
        self.relation_embeddings: Optional[torch.Tensor] = None
        self.entity_2_id: Optional[Dict[str, int]] = None
        self.relation_2_id: Optional[Dict[str, int]] = None
        self.id_2_entity: Optional[Dict[int, str]] = None
        self.id_2_relation: Optional[Dict[int, str]] = None
        self.entity_faiss_index: Optional[Any] = None # For FAISS index object

    def _load_sentence_transformer_model(self, model_name: str) -> None:
        """Loads or re-loads the SentenceTransformer model if necessary."""
        if self.model is not None and self.model_name == model_name:
            logger.debug(f"SentenceTransformer model '{model_name}' is already loaded.")
            return

        logger.info(f"Loading SentenceTransformer model: '{model_name}' onto device '{self.device}'...")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model_name = model_name
            logger.info(f"Successfully loaded model '{self.model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}", exc_info=True)
            self.model = None
            self.model_name = None
            raise

    def _ensure_model_and_embeddings_initialized(self, check_entity_embeddings: bool = False, check_relation_embeddings: bool = False, check_faiss_index: bool = False) -> bool:
        """Checks if the model and specified embeddings/indexes are initialized."""
        if not self.model or not self.model_name:
            logger.error("SentenceTransformer model not loaded. Call initialize_embeddings() with a model_name first.")
            return False
        if check_entity_embeddings and (self.entity_embeddings is None or self.entity_2_id is None or self.id_2_entity is None):
            logger.error("Entity embeddings and/or mappings are not initialized. Call initialize_embeddings() first.")
            return False
        if check_relation_embeddings and (self.relation_embeddings is None or self.relation_2_id is None or self.id_2_relation is None):
            logger.error("Relation embeddings and/or mappings are not initialized. Call initialize_embeddings() first.")
            return False
        if check_faiss_index and FAISS_AVAILABLE and self.entity_faiss_index is None:
            if self.entity_embeddings is not None: 
                logger.error("FAISS index for entities is not initialized, though entity embeddings might exist. Call initialize_embeddings() again or check FAISS build process.")
                return False
        return True

    def _process_sample(self, sample: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        try:
            graph_data = sample.get('graph')
            if not graph_data or not isinstance(graph_data, list):
                return []
            
            valid_triples: List[Tuple[str, str, str]] = []
            for triple in graph_data:
                if isinstance(triple, (list, tuple)) and len(triple) == 3 and \
                   all(isinstance(item, str) and item.strip() for item in triple): 
                    valid_triples.append(tuple(s.strip() for s in triple)) 
                else:
                    logger.warning(f"Skipping malformed or empty component triple: {triple} in sample.")
            return valid_triples
        except Exception as e:
            logger.error(f"Error processing sample data: {sample}. Error: {e}", exc_info=True)
            return []

    def initialize_embeddings(self, 
                              model_name: str, 
                              embedding_encode_batch_size: int = 1024):
        """
        Initializes embeddings for ALL entities and relations in memory 
        by fetching them directly from the Neo4j database.
        No disk caching is performed by this method.
        """
        logger.info(f"Initializing in-memory embeddings for ALL DB items using model='{model_name}' on device {self.device}.")
        
        try:
            self._load_sentence_transformer_model(model_name)
        except Exception: 
            logger.error(f"Cannot initialize embeddings due to model loading failure for '{model_name}'.")
            return

        logger.info(f"Computing embeddings and mappings directly on {self.device}...")
        # Call _compute_embeddings_in_memory without entity/relation source lists
        self._compute_embeddings_in_memory(embedding_encode_batch_size) 
        
        logger.info("In-memory embeddings initialization complete.")

    def _compute_embeddings_in_memory(self, 
                                      embedding_encode_batch_size: int):
        if self.model is None:
            logger.error("SentenceTransformer model not loaded. Cannot compute embeddings.")
            return

        # Always fetch entities and relations from the Neo4j database
        logger.info("Fetching all entity IDs from Neo4j...")
        entity_ids_to_process = self.get_all_entities()
        
        logger.info("Fetching all relation types from Neo4j...")
        relation_types_to_process = self.get_all_relations()

        if entity_ids_to_process: 
            logger.info(f"Generating embeddings for {len(entity_ids_to_process)} unique, non-null entities...")
            unique_entity_ids = sorted(list(set(entity_ids_to_process))) 
            if not all(unique_entity_ids): 
                logger.warning("Empty string found in unique_entity_ids after processing. Filtering them out.")
                unique_entity_ids = [eid for eid in unique_entity_ids if eid] 

            if unique_entity_ids:
                logger.info(f"Computing entity embeddings directly on {self.device}...")
                self.entity_embeddings = self.model.encode(
                    unique_entity_ids, 
                    batch_size=embedding_encode_batch_size, 
                    convert_to_tensor=True, 
                    show_progress_bar=True,
                    device=self.device
                )
                self.entity_2_id = {ent: i for i, ent in enumerate(unique_entity_ids)}
                self.id_2_entity = {i: ent for i, ent in enumerate(unique_entity_ids)}

                if FAISS_AVAILABLE and self.entity_embeddings is not None and self.entity_embeddings.numel() > 0 :
                    logger.info("Building FAISS index for entity embeddings in memory...")
                    try:
                        cpu_embeddings = self.entity_embeddings.cpu().numpy()
                        dimension = cpu_embeddings.shape[1]
                        index = faiss.IndexFlatIP(dimension) # Using Inner Product similarity
                        index.add(cpu_embeddings)
                        self.entity_faiss_index = index 
                        logger.info(f"In-memory FAISS index built (ntotal={index.ntotal}).")
                    except Exception as e:
                        logger.error(f"Failed to build in-memory FAISS index: {e}", exc_info=True)
                        self.entity_faiss_index = None
                elif self.entity_embeddings is None or self.entity_embeddings.numel() == 0:
                        logger.info("Skipping FAISS index creation as entity embeddings are empty.")
                        self.entity_faiss_index = None
            else: 
                logger.warning("No valid non-empty entity IDs left after filtering to generate embeddings.")
                self.entity_embeddings, self.entity_2_id, self.id_2_entity, self.entity_faiss_index = None, {}, {}, None
        else: 
            logger.info("No entity IDs found in DB; entity embeddings and mappings will be empty/None.")
            self.entity_embeddings, self.entity_2_id, self.id_2_entity, self.entity_faiss_index = None, {}, {}, None
        
        if relation_types_to_process: 
            logger.info(f"Generating embeddings for {len(relation_types_to_process)} unique, non-null relations...")
            unique_relation_types = sorted(list(set(relation_types_to_process)))
            if not all(unique_relation_types):
                logger.warning("Empty string found in unique_relation_types after processing. Filtering them out.")
                unique_relation_types = [rel for rel in unique_relation_types if rel]

            if unique_relation_types:
                logger.info(f"Computing relation embeddings directly on {self.device}...")
                self.relation_embeddings = self.model.encode(
                    unique_relation_types, 
                    batch_size=embedding_encode_batch_size, 
                    convert_to_tensor=True, 
                    show_progress_bar=True,
                    device=self.device 
                )
                self.relation_2_id = {rel: i for i, rel in enumerate(unique_relation_types)}
                self.id_2_relation = {i: rel for i, rel in enumerate(unique_relation_types)}
            else: 
                logger.warning("No valid non-empty relation types left after filtering to generate embeddings.")
                self.relation_embeddings, self.relation_2_id, self.id_2_relation = None, {}, {}
        else: 
            logger.info("No relation types found in DB; relation embeddings and mappings will be empty/None.")
            self.relation_embeddings, self.relation_2_id, self.id_2_relation = None, {}, {}

    def create_indexes(self):
        try:
            with self.driver.session() as session:
                session.run("CREATE INDEX entity_id_index IF NOT EXISTS FOR (n:ENTITY) ON (n.id)")
                # Corrected: Relation type index on RELATIONSHIP type, not node label
                session.run("CREATE INDEX relation_type_index IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)")
                logger.info("Database indexes ensured.")
        except Exception as e:
            logger.error(f"Failed to create or ensure indexes: {e}", exc_info=True)

    def close(self):
        if self.driver:
            logger.info("Closing Neo4j driver connection.")
            self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            logger.info("Clearing database (MATCH (n) DETACH DELETE n)...")
            try:
                result = session.run("MATCH (n) RETURN count(n) as count").single()
                if result and result["count"] > 0:
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.info("Database cleared successfully.")
                else:
                    logger.info("Database is already empty or no nodes found.")
            except Exception as e:
                logger.error(f"Failed to clear database: {e}", exc_info=True)

    def _batch_process_triples_to_neo4j(self, session, triples: List[Tuple[str, str, str]], batch_size: int = 500):
        def _run_batch_tx(tx, batch_data: List[Tuple[str, str, str]]):
            query = """
            UNWIND $batch as triple
            MERGE (head:ENTITY {id: triple[0]})
            MERGE (tail:ENTITY {id: triple[2]})
            MERGE (head)-[r:RELATION {type: triple[1]}]->(tail)
            """
            tx.run(query, batch=batch_data)

        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            if batch:
                try:
                    session.execute_write(_run_batch_tx, batch_data=batch)
                except Exception as e:
                    logger.error(f"Error processing batch of triples (index {i} to {i+batch_size-1}): {e}", exc_info=True)

    def load_graph_from_dataset(self, 
                                  input_source: str, 
                                  model_name_for_embeddings: str,
                                  batch_size_neo4j: int = 500, 
                                  embedding_encode_batch_size: int = 1024,
                                  hf_dataset_split: Optional[str] = None
                                  ):
        """Load graph data from a dataset, load into Neo4j, and initialize in-memory embeddings for ALL items in the DB."""
        logger.info(f"Loading dataset from source '{input_source}' for KG structure.")
        
        dataset_iterable: List[Dict[str, Any]]
        # ... (dataset loading logic - kept as is)
        if os.path.exists(input_source): 
            if input_source.endswith(".jsonl"):
                dataset_iterable = []
                try:
                    with open(input_source, 'r', encoding='utf-8') as f:
                        for line in f: dataset_iterable.append(json.loads(line))
                except Exception as e:
                    logger.error(f"Failed to load JSONL file '{input_source}': {e}", exc_info=True); return
            elif input_source.endswith(".json"):
                try:
                    with open(input_source, 'r', encoding='utf-8') as f: dataset_iterable = json.load(f)
                    if not isinstance(dataset_iterable, list):
                        logger.error(f"JSON file '{input_source}' does not contain a list."); return
                except Exception as e:
                    logger.error(f"Failed to load JSON file '{input_source}': {e}", exc_info=True); return
            else: 
                try:
                    from datasets import load_from_disk # type: ignore
                    loaded_hf_dataset = load_from_disk(input_source)
                    dataset_iterable = list(loaded_hf_dataset) # type: ignore
                except Exception as e:
                    logger.error(f"Failed to load dataset from disk directory '{input_source}': {e}. If it's a HF Hub ID, ensure hf_dataset_split is set.", exc_info=True); return
        else: 
            if not hf_dataset_split:
                logger.error(f"'{input_source}' is not a local file/dir. 'hf_dataset_split' must be provided for Hugging Face Hub datasets.")
                return
            from datasets import load_dataset as hf_load_dataset # type: ignore
            try:
                loaded_hf_dataset = hf_load_dataset(input_source, split=hf_dataset_split)
                dataset_iterable = list(loaded_hf_dataset) # type: ignore
            except Exception as e:
                logger.error(f"Failed to load Hugging Face Hub dataset '{input_source}' (split '{hf_dataset_split}'): {e}", exc_info=True)
                return

        logger.info(f"Dataset loaded: {len(dataset_iterable)} samples to process for triples.")
        
        all_triples_to_load: List[Tuple[str, str, str]] = []
        for sample in tqdm(dataset_iterable, desc="Extracting triples"):
            triples = self._process_sample(sample)
            if triples: all_triples_to_load.extend(triples)
        
        if all_triples_to_load:
            unique_triples_to_load = sorted(list(set(all_triples_to_load))) 
            logger.info(f"Extracted {len(unique_triples_to_load)} unique triples. Loading into Neo4j (batch size: {batch_size_neo4j})...")
            with self.driver.session() as session:
                self._batch_process_triples_to_neo4j(session, unique_triples_to_load, batch_size_neo4j)
            logger.info("Finished loading triples into Neo4j.")
        else:
            logger.warning("No triples extracted from the dataset. Neo4j loading skipped.")

        # Now initialize embeddings based on whatever is in the DB.
        self.initialize_embeddings(
            model_name=model_name_for_embeddings,
            embedding_encode_batch_size=embedding_encode_batch_size
        )
        # No longer passing dataset_entity_ids or dataset_relation_types to initialize_embeddings
        # The initialize_embeddings method will now always call get_all_entities/relations.

        try: 
            with self.driver.session() as session:
                stats = session.run("""
                    MATCH (n:ENTITY) WITH count(n) as entity_count
                    MATCH ()-[r:RELATION]->() WITH entity_count, count(DISTINCT r) as rel_count
                    RETURN entity_count, rel_count
                """).single()
                if stats: logger.info(f"DB stats post-load and embedding init: Entities: {stats['entity_count']}, Relationships: {stats['rel_count']}")
        except Exception as e: logger.warning(f"Could not retrieve DB stats: {e}")
    
    # --- Other methods (get_shortest_paths, get_target_entities, etc.) remain unchanged ---
    def get_shortest_paths(self, source_id: str, target_id: str, max_depth: int = 5) -> List[List[Tuple[str, str, str]]]:
        paths_result: List[List[Tuple[str, str, str]]] = []
        if not source_id or not target_id:
            logger.warning("Source ID or Target ID is empty for get_shortest_paths.")
            return paths_result
        if max_depth <= 0: 
            logger.warning("max_depth must be at least 1 for paths with relationships in get_shortest_paths.")
            if source_id == target_id:
                logger.debug(f"Source and target nodes are the same ('{source_id}') and max_depth <= 0. Returning a 0-length path representation.")
                paths_result.append([]) 
                return paths_result
            return paths_result 
            
        if source_id == target_id:
            logger.debug(f"Source and target nodes are the same ('{source_id}'). Returning a 0-length path representation.")
            paths_result.append([]) 
            return paths_result

        query = f"""
        MATCH (source:ENTITY {{id: $source_id}}), (target:ENTITY {{id: $target_id}})
        MATCH k_paths = allShortestPaths((source)-[:RELATION*1..{max_depth}]->(target))
        RETURN k_paths
        """
        try:
            with self.driver.session() as session:
                results = session.run(query, source_id=source_id, target_id=target_id)
                for record in results:
                    path_obj = record["k_paths"]
                    path_info_tuples: List[Tuple[str, str, str]] = []
                    
                    if not path_obj.nodes or not path_obj.relationships: 
                        logger.warning(f"Unexpected empty path object for distinct source/target: {path_obj}")
                        continue

                    valid_path_segment = True
                    for i, rel_obj in enumerate(path_obj.relationships):
                        start_node = path_obj.nodes[i] 
                        end_node = path_obj.nodes[i+1]   
                        
                        head_id = start_node.get('id')
                        tail_id = end_node.get('id')
                        relation_type = rel_obj.get('type')

                        if head_id is None or tail_id is None or relation_type is None:
                            logger.warning(f"Path segment contains node/relationship with missing id/type in path: {path_obj}")
                            valid_path_segment = False; break 
                        path_info_tuples.append((head_id, relation_type, tail_id))
                    
                    if valid_path_segment and path_info_tuples: 
                        paths_result.append(path_info_tuples)
        except Exception as e:
            logger.error(f"Generic error getting shortest paths for '{source_id}' -> '{target_id}': {e}", exc_info=True)
        return paths_result

    def get_target_entities(self, source_id: str, relation_type: str, direction: str = "out") -> List[str]:
        if direction not in ["in", "out"]:
            logger.error(f"Invalid direction '{direction}'. Must be 'in' or 'out'."); return []
        query_template = (
            "MATCH (source:ENTITY {id: $source_id})-[r:RELATION {type: $relation_type}]->(target:ENTITY) RETURN DISTINCT target.id as target_id"
            if direction == "out" else
            "MATCH (target:ENTITY)-[r:RELATION {type: $relation_type}]->(source:ENTITY {id: $source_id}) RETURN DISTINCT target.id as target_id"
        )
        try:
            with self.driver.session() as session:
                result = session.run(query_template, source_id=source_id, relation_type=relation_type)
                return [record["target_id"] for record in result if record["target_id"]]
        except Exception as e:
            logger.error(f"Error getting target entities (src='{source_id}', rel='{relation_type}', dir='{direction}'): {e}", exc_info=True); return []
            
    def get_related_relations(self, entity_id: str, direction: str = "out") -> List[str]:
        if direction not in ["in", "out"]:
            logger.error(f"Invalid direction '{direction}'. Must be 'in' or 'out'."); return []
        query_template = (
            "MATCH (n:ENTITY {id: $entity_id})-[r:RELATION]->() WHERE r.type IS NOT NULL RETURN DISTINCT r.type as relation_type"
            if direction == "out" else
            "MATCH ()-[r:RELATION]->(n:ENTITY {id: $entity_id}) WHERE r.type IS NOT NULL RETURN DISTINCT r.type as relation_type"
        )
        try:
            with self.driver.session() as session:
                result = session.run(query_template, entity_id=entity_id)
                return [record["relation_type"] for record in result if record["relation_type"]]
        except Exception as e:
            logger.error(f"Error getting related relations for entity '{entity_id}', direction '{direction}': {e}", exc_info=True); return []

    def get_all_entities(self) -> List[str]:
        """Gets all unique entity IDs currently in the Neo4j database."""
        with self.driver.session() as session:
            query = "MATCH (n:ENTITY) WHERE n.id IS NOT NULL RETURN DISTINCT n.id AS entity"
            try:
                result = session.run(query); return [record["entity"] for record in result if record["entity"]]
            except Exception as e: logger.error(f"Error getting all entities: {e}", exc_info=True); return []

    def get_all_relations(self) -> List[str]:
        """Gets all unique relation types currently in the Neo4j database."""
        with self.driver.session() as session:
            query = "MATCH ()-[r:RELATION]->() WHERE r.type IS NOT NULL RETURN DISTINCT r.type AS relation"
            try:
                result = session.run(query); return [record["relation"] for record in result if record["relation"]]
            except Exception as e: logger.error(f"Error getting all relations: {e}", exc_info=True); return []

    def get_related_relations_by_question(self, entity_id: str, question: str) -> List[Tuple[str, float]]:
        if not self._ensure_model_and_embeddings_initialized(check_relation_embeddings=True): return []
        if self.relation_embeddings is None or self.relation_2_id is None or self.model is None or not self.relation_2_id:
            if not (self.relation_embeddings is None and not self.relation_2_id): 
                logger.warning(f"Relation embeddings/mappings not fully available for semantic search for entity '{entity_id}'. Relation_2_id empty: {not self.relation_2_id}, Embeddings None: {self.relation_embeddings is None}")
            return []
        
        try:
            question_emb = self.model.encode(question, convert_to_tensor=True, show_progress_bar=False) 
            question_emb = question_emb.to(self.relation_embeddings.device) 
        except Exception as e: logger.error(f"Error encoding question '{question}': {e}", exc_info=True); return []

        outgoing_relations = self.get_related_relations(entity_id, direction="out")
        if not outgoing_relations: return []

        valid_rel_indices, valid_rels_for_scoring = [], []
        for rel_type in outgoing_relations:
            if self.relation_2_id and rel_type in self.relation_2_id: 
                valid_rel_indices.append(self.relation_2_id[rel_type]) 
                valid_rels_for_scoring.append(rel_type)
        
        if not valid_rel_indices: 
            logger.debug(f"No relations for entity '{entity_id}' found in precomputed embeddings (no overlap with relation_2_id).")
            return []
        
        related_embs_subset = self.relation_embeddings[torch.tensor(valid_rel_indices, device=self.relation_embeddings.device)] 

        try:
            if question_emb.numel() == 0 or related_embs_subset.numel() == 0:
                logger.warning("Cannot compute similarity with empty question or relation embeddings.")
                return []
            similarities = self.model.similarity(question_emb, related_embs_subset)[0] # type: ignore
            
            scores, indices_in_subset = torch.sort(similarities, descending=True)
            
            return [(valid_rels_for_scoring[idx.item()], score.item()) for score, idx in zip(scores, indices_in_subset)]
        except Exception as e: logger.error(f"Error computing/sorting similarities for entity '{entity_id}': {e}", exc_info=True); return []

    def get_related_entities_by_question(self, question: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self._ensure_model_and_embeddings_initialized(check_entity_embeddings=True, check_faiss_index=FAISS_AVAILABLE): return []
        if self.entity_embeddings is None or self.id_2_entity is None or self.model is None or not self.id_2_entity:
            if not (self.entity_embeddings is None and not self.id_2_entity):
                logger.warning(f"Entity embeddings/mappings not fully available for semantic entity search. Id_2_entity empty: {not self.id_2_entity}, Embeddings None: {self.entity_embeddings is None}")
            return []
        if top_k <=0: logger.warning("top_k must be positive for get_related_entities_by_question."); return []

        try:
            question_emb = self.model.encode(question, convert_to_tensor=True, show_progress_bar=False)
            question_emb_np = question_emb.cpu().numpy().reshape(1, -1) 
        except Exception as e: logger.error(f"Error encoding question '{question}': {e}", exc_info=True); return []
        
        results: List[Tuple[str, float]] = []
        try:
            if FAISS_AVAILABLE and self.entity_faiss_index is not None:
                logger.debug(f"Searching with FAISS index for top {top_k} entities.")
                if self.entity_faiss_index.ntotal == 0:
                    logger.warning("FAISS index is empty. Cannot search.")
                    return []
                actual_k_faiss = min(top_k, self.entity_faiss_index.ntotal)
                if actual_k_faiss == 0: return []

                scores_np, indices_np = self.entity_faiss_index.search(question_emb_np, actual_k_faiss)
                
                for i in range(indices_np.shape[1]):
                    faiss_idx = indices_np[0, i]
                    score = scores_np[0, i]
                    if faiss_idx != -1: # FAISS can return -1 for invalid indices
                        entity_name = self.id_2_entity.get(faiss_idx) 
                        if entity_name:
                            results.append((entity_name, float(score))) 
            
            elif not FAISS_AVAILABLE: 
                logger.warning("FAISS not available. Falling back to slow exact similarity search for entities. This can be very slow for large KGs.")
                if self.entity_embeddings.numel() == 0 : # type: ignore 
                    logger.warning("Entity embeddings are empty. Cannot perform similarity search.")
                    return []
                
                similarities = self.model.similarity(question_emb.to(self.entity_embeddings.device), self.entity_embeddings)[0] # type: ignore
                
                actual_k = min(top_k, similarities.size(0))
                if actual_k == 0: return []

                scores, indices = torch.topk(similarities, k=actual_k, largest=True)
                for score, idx in zip(scores, indices):
                    entity_name = self.id_2_entity.get(idx.item()) 
                    if entity_name: results.append((entity_name, score.item()))
            else: 
                logger.error("FAISS is available but index not loaded/built, or entity embeddings are missing. Cannot perform fast entity search.")
                return [] 
            return results
        except Exception as e: logger.error(f"Error computing/sorting similarities for entity embeddings: {e}", exc_info=True); return []