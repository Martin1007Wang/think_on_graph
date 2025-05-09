import torch
import pickle
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
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
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
            logger.error("FAISS index for entities is not initialized. Call initialize_embeddings() first.")
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
                   all(isinstance(item, str) and item.strip() for item in triple): # Ensure non-empty strings
                    valid_triples.append(tuple(s.strip() for s in triple)) 
                else:
                    logger.warning(f"Skipping malformed or empty component triple: {triple} in sample.")
            return valid_triples
        except Exception as e:
            logger.error(f"Error processing sample data: {sample}. Error: {e}", exc_info=True)
            return []

    def initialize_embeddings(self, 
                              dataset_name_for_caching: str, 
                              split_name_for_caching: str, 
                              model_name: str, 
                              force_recompute: bool = False,
                              embedding_encode_batch_size: int = 1024,
                              entity_source_list: Optional[List[str]] = None,
                              relation_source_list: Optional[List[str]] = None):
        """
        Initializes embeddings for entities and relations.
        Embeddings are loaded from cache if available, or computed and saved.
        Entity and relation sources can be provided directly to scope embedding generation.
        """
        logger.info(f"Initializing embeddings for cache key: dataset='{dataset_name_for_caching}', split='{split_name_for_caching}' using model='{model_name}'.")
        
        try:
            self._load_sentence_transformer_model(model_name)
        except Exception: 
            logger.error(f"Cannot initialize embeddings due to model loading failure for '{model_name}'.")
            return

        emb_dir = os.path.join("embeddings", dataset_name_for_caching, split_name_for_caching, self.model_name) # type: ignore
        os.makedirs(emb_dir, exist_ok=True)
        logger.info(f"Embeddings cache directory: {os.path.abspath(emb_dir)}")

        paths = {
            "entity_emb": os.path.join(emb_dir, "entity_embeddings.pt"),
            "entity_2_id": os.path.join(emb_dir, "entity_2_id.pkl"),
            "id_2_entity": os.path.join(emb_dir, "id_2_entity.pkl"),
            "relation_emb": os.path.join(emb_dir, "relation_embeddings.pt"),
            "relation_2_id": os.path.join(emb_dir, "relation_2_id.pkl"),
            "id_2_relation": os.path.join(emb_dir, "id_2_relation.pkl"),
            "entity_faiss_index": os.path.join(emb_dir, "entity_faiss.index") # Path for FAISS index
        }
        
        # Check for all files, including FAISS index if FAISS is available
        files_to_check = list(paths.values())
        if not FAISS_AVAILABLE:
            files_to_check.remove(paths["entity_faiss_index"])
        all_files_exist = all(os.path.exists(p) for p in files_to_check)
        
        if not force_recompute and all_files_exist:
            logger.info("Loading precomputed embeddings, mappings, and FAISS index (if applicable) from disk...")
            try:
                self.entity_embeddings = torch.load(paths["entity_emb"], map_location=self.device)
                with open(paths["entity_2_id"], 'rb') as f: self.entity_2_id = pickle.load(f)
                with open(paths["id_2_entity"], 'rb') as f: self.id_2_entity = pickle.load(f)
                
                self.relation_embeddings = torch.load(paths["relation_emb"], map_location=self.device)
                with open(paths["relation_2_id"], 'rb') as f: self.relation_2_id = pickle.load(f)
                with open(paths["id_2_relation"], 'rb') as f: self.id_2_relation = pickle.load(f)

                if FAISS_AVAILABLE and os.path.exists(paths["entity_faiss_index"]):
                    self.entity_faiss_index = faiss.read_index(paths["entity_faiss_index"])
                    logger.info("FAISS index loaded successfully.")
                elif FAISS_AVAILABLE and not os.path.exists(paths["entity_faiss_index"]):
                     logger.warning(f"FAISS index file missing at {paths['entity_faiss_index']}. Will attempt recomputation if other files are also missing or force_recompute is True.")
                     # This will lead to recomputation if all_files_exist was false due to this
                
                logger.info("Successfully loaded embeddings and mappings from disk.")
            except Exception as e:
                logger.warning(f"Error loading from disk (files might be corrupted): {e}. Attempting recomputation.", exc_info=True)
                self._compute_and_save_embeddings(paths, embedding_encode_batch_size, entity_source_list, relation_source_list)
        else:
            if force_recompute:
                logger.info("Forcing recomputation of embeddings.")
            else:
                logger.info("Embeddings/mappings not found on disk or some files are missing. Computing new ones.")
            self._compute_and_save_embeddings(paths, embedding_encode_batch_size, entity_source_list, relation_source_list)
        
        if self.entity_embeddings is not None:
            self.entity_embeddings = self.entity_embeddings.to(self.device)
        if self.relation_embeddings is not None:
            self.relation_embeddings = self.relation_embeddings.to(self.device)

    def _compute_and_save_embeddings(self, 
                                     paths: Dict[str,str], 
                                     embedding_encode_batch_size: int,
                                     entity_source_list: Optional[List[str]] = None,
                                     relation_source_list: Optional[List[str]] = None):
        if self.model is None:
            logger.error("SentenceTransformer model not loaded. Cannot compute embeddings.")
            return

        # Determine entity_ids and relation_types
        entity_ids = entity_source_list
        if entity_ids is None:
            logger.info("Fetching all entity IDs from Neo4j as no source list provided...")
            entity_ids = self.get_all_entities() # Fallback to fetching all from DB
        
        relation_types = relation_source_list
        if relation_types is None:
            logger.info("Fetching all relation types from Neo4j as no source list provided...")
            relation_types = self.get_all_relations() # Fallback to fetching all from DB

        if entity_ids:
            logger.info(f"Generating embeddings for {len(entity_ids)} unique, non-null entities...")
            # Ensure entity_ids are unique before encoding, if they came from triples they might not be
            unique_entity_ids = sorted(list(set(entity_ids)))
            self.entity_embeddings = self.model.encode(unique_entity_ids, batch_size=embedding_encode_batch_size, convert_to_tensor=True, show_progress_bar=True)
            self.entity_2_id = {ent: i for i, ent in enumerate(unique_entity_ids)}
            self.id_2_entity = {i: ent for i, ent in enumerate(unique_entity_ids)}
            
            logger.info(f"Saving entity embeddings and mappings to {os.path.dirname(paths['entity_emb'])}")
            torch.save(self.entity_embeddings, paths["entity_emb"])
            with open(paths["entity_2_id"], 'wb') as f: pickle.dump(self.entity_2_id, f)
            with open(paths["id_2_entity"], 'wb') as f: pickle.dump(self.id_2_entity, f)

            if FAISS_AVAILABLE and self.entity_embeddings is not None:
                logger.info("Building FAISS index for entity embeddings...")
                try:
                    dimension = self.entity_embeddings.shape[1]
                    # Using IndexFlatIP because SentenceTransformer embeddings are often normalized,
                    # making inner product equivalent to cosine similarity.
                    # For unnormalized embeddings or explicit cosine, IndexFlatL2 + normalization is an option.
                    index = faiss.IndexFlatIP(dimension) 
                    
                    # If using GPU for FAISS index (requires faiss-gpu and GPU resources)
                    # if self.device == "cuda":
                    #     res = faiss.StandardGpuResources()
                    #     index = faiss.index_cpu_to_gpu(res, 0, index) # 0 is the GPU id

                    index.add(self.entity_embeddings.cpu().numpy()) # FAISS typically expects numpy on CPU for add
                    faiss.write_index(index, paths["entity_faiss_index"])
                    self.entity_faiss_index = index # Keep the CPU index in memory
                    logger.info(f"FAISS index built and saved to {paths['entity_faiss_index']}")
                except Exception as e:
                    logger.error(f"Failed to build or save FAISS index: {e}", exc_info=True)
                    self.entity_faiss_index = None # Ensure it's None if building failed

        else:
            logger.warning("No valid entity IDs found/provided to generate embeddings.")
            self.entity_embeddings, self.entity_2_id, self.id_2_entity, self.entity_faiss_index = None, None, None, None
        
        if relation_types:
            logger.info(f"Generating embeddings for {len(relation_types)} unique, non-null relations...")
            unique_relation_types = sorted(list(set(relation_types)))
            self.relation_embeddings = self.model.encode(unique_relation_types, batch_size=embedding_encode_batch_size, convert_to_tensor=True, show_progress_bar=True)
            self.relation_2_id = {rel: i for i, rel in enumerate(unique_relation_types)}
            self.id_2_relation = {i: rel for i, rel in enumerate(unique_relation_types)}

            logger.info(f"Saving relation embeddings and mappings to {os.path.dirname(paths['relation_emb'])}")
            torch.save(self.relation_embeddings, paths["relation_emb"])
            with open(paths["relation_2_id"], 'wb') as f: pickle.dump(self.relation_2_id, f)
            with open(paths["id_2_relation"], 'wb') as f: pickle.dump(self.id_2_relation, f)
        else:
            logger.warning("No valid relation types found/provided to generate embeddings.")
            self.relation_embeddings, self.relation_2_id, self.id_2_relation = None, None, None

    def create_indexes(self):
        try:
            with self.driver.session() as session:
                session.run("CREATE INDEX entity_id_index IF NOT EXISTS FOR (n:ENTITY) ON (n.id)")
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
                                dataset_name_for_caching: str, 
                                split_name_for_caching: str,  
                                model_name_for_embeddings: str,
                                batch_size_neo4j: int = 500, 
                                embedding_encode_batch_size: int = 1024, # Added for configurability
                                hf_dataset_split: Optional[str] = None,
                                force_recompute_embeddings: bool = False # Added
                                ):
        """Load graph data from a dataset, load into Neo4j, and initialize embeddings for this dataset."""
        logger.info(f"Loading dataset from source '{input_source}' for KG structure.")
        
        dataset_iterable: List[Dict[str, Any]]
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
        
        dataset_entity_ids: Optional[List[str]] = None
        dataset_relation_types: Optional[List[str]] = None

        if all_triples_to_load:
            unique_triples_to_load = sorted(list(set(all_triples_to_load))) 
            logger.info(f"Extracted {len(unique_triples_to_load)} unique triples. Loading into Neo4j (batch size: {batch_size_neo4j})...")
            with self.driver.session() as session:
                self._batch_process_triples_to_neo4j(session, unique_triples_to_load, batch_size_neo4j)
            logger.info("Finished loading triples into Neo4j.")

            # Extract unique entities and relations FROM THE LOADED DATASET for scoped embedding generation
            dataset_entities_set = set()
            dataset_relations_set = set()
            for h, r, t in unique_triples_to_load:
                dataset_entities_set.add(h)
                dataset_entities_set.add(t)
                dataset_relations_set.add(r)
            dataset_entity_ids = sorted(list(dataset_entities_set))
            dataset_relation_types = sorted(list(dataset_relations_set))
            logger.info(f"Derived {len(dataset_entity_ids)} unique entities and {len(dataset_relation_types)} unique relations from the input dataset for embedding.")

        else:
            logger.warning("No triples extracted from the dataset. Neo4j loading and embedding initialization might be skipped or use existing DB state if not scoped.")

        try: 
            with self.driver.session() as session:
                stats = session.run("""
                    MATCH (n:ENTITY) WITH count(n) as entity_count
                    MATCH ()-[r:RELATION]->() WITH entity_count, count(DISTINCT r) as rel_count
                    RETURN entity_count, rel_count
                """).single()
                if stats: logger.info(f"DB stats post-load: Entities: {stats['entity_count']}, Relationships: {stats['rel_count']}")
        except Exception as e: logger.warning(f"Could not retrieve DB stats: {e}")
        
        self.initialize_embeddings(dataset_name_for_caching, 
                                   split_name_for_caching, 
                                   model_name=model_name_for_embeddings,
                                   force_recompute=force_recompute_embeddings,
                                   embedding_encode_batch_size=embedding_encode_batch_size,
                                   entity_source_list=dataset_entity_ids,
                                   relation_source_list=dataset_relation_types)

    def get_shortest_paths(self, source_id: str, target_id: str, max_depth: int = 5) -> List[List[Tuple[str, str, str]]]:
        paths_result: List[List[Tuple[str, str, str]]] = []
        if not source_id or not target_id:
            logger.warning("Source ID or Target ID is empty for get_shortest_paths.")
            return paths_result
        if max_depth <= 0: # Cypher *1..N requires N >= 1
            logger.warning("max_depth must be at least 1 for paths with relationships in get_shortest_paths.")
            # If source_id == target_id and max_depth is 0, a 0-length path is still valid.
            # However, the Cypher query below is for paths of length 1 to max_depth.
            if source_id == target_id:
                logger.debug(f"Source and target nodes are the same ('{source_id}') and max_depth <= 0. Returning a 0-length path representation.")
                paths_result.append([]) # Represent as an empty list of steps for the 0-length path
                return paths_result
            return paths_result # No paths with relationships if max_depth <= 0 and source != target
            
        # *** 主要修改：在执行查询前处理 source_id == target_id 的情况 ***
        if source_id == target_id:
            logger.debug(f"Source and target nodes are the same ('{source_id}'). Returning a 0-length path representation.")
            paths_result.append([]) # 表示一个0跳的路径 (一个空列表代表路径中的步骤)
            return paths_result

        # 仅当 source_id != target_id 时，才执行以下查询
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
                    
                    # path_obj.nodes 和 path_obj.relationships 应该存在，因为查询是 *1..max_depth
                    if not path_obj.nodes or not path_obj.relationships: 
                        # 对于 allShortestPaths((s)-[*1..N]->(t)) 且 s!=t, 
                        # 返回的路径对象总是会有节点和关系。
                        # 如果这里为空，可能是数据问题或意外的路径对象。
                        logger.warning(f"Unexpected empty path object for distinct source/target: {path_obj}")
                        continue

                    valid_path_segment = True
                    for i, rel_obj in enumerate(path_obj.relationships):
                        start_node = path_obj.nodes[i] 
                        end_node = path_obj.nodes[i+1]  
                        
                        head_id = start_node.get('id')
                        tail_id = end_node.get('id')
                        relation_type = rel_obj.get('type') # 你的数据模型中，关系类型存储在 type 属性里

                        if head_id is None or tail_id is None or relation_type is None:
                            logger.warning(f"Path segment contains node/relationship with missing id/type in path: {path_obj}")
                            valid_path_segment = False; break # Discard this malformed path
                        path_info_tuples.append((head_id, relation_type, tail_id))
                    
                    if valid_path_segment and path_info_tuples: # Ensure path_info_tuples is not empty
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
        if self.relation_embeddings is None or self.relation_2_id is None or self.model is None: # Should be caught by ensure, but defensive
             logger.error("Relation embeddings/mappings or model not available for semantic search.")
             return []
        
        try:
            question_emb = self.model.encode(question, convert_to_tensor=True, show_progress_bar=False) 
            question_emb = question_emb.to(self.relation_embeddings.device) 
        except Exception as e: logger.error(f"Error encoding question '{question}': {e}", exc_info=True); return []

        outgoing_relations = self.get_related_relations(entity_id, direction="out")
        if not outgoing_relations: return []

        valid_rel_indices, valid_rels_for_scoring = [], []
        for rel_type in outgoing_relations:
            if rel_type in self.relation_2_id: 
                valid_rel_indices.append(self.relation_2_id[rel_type]) 
                valid_rels_for_scoring.append(rel_type)
        
        if not valid_rel_indices: 
            logger.debug(f"No relations for entity '{entity_id}' found in precomputed embeddings.")
            return []
        
        # Ensure tensors are on the same device before similarity calculation
        related_embs_subset = self.relation_embeddings[torch.tensor(valid_rel_indices, device=self.relation_embeddings.device)] 

        try:
            similarities = self.model.similarity(question_emb, related_embs_subset)[0] # type: ignore
            # pytorch_cos_sim returns a 2D tensor [query_count, corpus_count], so take [0]
            
            scores, indices_in_subset = torch.sort(similarities, descending=True)
            
            return [(valid_rels_for_scoring[idx.item()], score.item()) for score, idx in zip(scores, indices_in_subset)]
        except Exception as e: logger.error(f"Error computing/sorting similarities for entity '{entity_id}': {e}", exc_info=True); return []

    def get_related_entities_by_question(self, question: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self._ensure_model_and_embeddings_initialized(check_entity_embeddings=True, check_faiss_index=FAISS_AVAILABLE): return []
        if self.entity_embeddings is None or self.id_2_entity is None or self.model is None: # Defensive
            logger.error("Entity embeddings/mappings or model not available for semantic search.")
            return []
        if top_k <=0: logger.warning("top_k must be positive for get_related_entities_by_question."); return []

        try:
            question_emb = self.model.encode(question, convert_to_tensor=True, show_progress_bar=False)
            question_emb_np = question_emb.cpu().numpy().reshape(1, -1) # FAISS expects 2D numpy array
        except Exception as e: logger.error(f"Error encoding question '{question}': {e}", exc_info=True); return []
        
        results: List[Tuple[str, float]] = []
        try:
            if FAISS_AVAILABLE and self.entity_faiss_index is not None:
                logger.debug(f"Searching with FAISS index for top {top_k} entities.")
                # FAISS search returns distances (D) and indices (I)
                # For IndexFlatIP, higher scores (dot products) are better. FAISS returns them directly.
                scores_np, indices_np = self.entity_faiss_index.search(question_emb_np, top_k)
                
                for i in range(indices_np.shape[1]): # Iterate through found items for the first (only) query
                    faiss_idx = indices_np[0, i]
                    score = scores_np[0, i]
                    if faiss_idx != -1: # -1 can indicate no more results or error
                        entity_name = self.id_2_entity.get(faiss_idx)
                        if entity_name:
                            results.append((entity_name, float(score))) 
            
            elif not FAISS_AVAILABLE: # Fallback or primary if FAISS not installed
                 logger.warning("FAISS not available. Falling back to slow exact similarity search for entities. This can be very slow for large KGs.")
                 similarities = self.model.similarity(question_emb.to(self.entity_embeddings.device), self.entity_embeddings)[0] # type: ignore
                 
                 actual_k = min(top_k, similarities.size(0))
                 if actual_k == 0: return []

                 scores, indices = torch.topk(similarities, k=actual_k, largest=True)
                 for score, idx in zip(scores, indices):
                     entity_name = self.id_2_entity.get(idx.item()) 
                     if entity_name: results.append((entity_name, score.item()))
            else: # FAISS available but index is None
                logger.error("FAISS is available but index not loaded/built. Cannot perform fast entity search.")
                return [] # Or fallback, but this indicates an issue in initialization

            return results
        except Exception as e: logger.error(f"Error computing/sorting similarities for entity embeddings: {e}", exc_info=True); return []