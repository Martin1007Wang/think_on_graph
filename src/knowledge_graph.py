import torch
import json
import os
import pickle # For saving/loading Python objects like dictionaries
from typing import List, Tuple, Dict, Optional, Any, Set
# from dataclasses import dataclass # Not used in this KG class, but might be in your scripts
from neo4j import GraphDatabase # type: ignore
from tqdm import tqdm # type: ignore
import logging
import math
import hashlib # For creating a simple hash for cache invalidation if needed

# Attempt to import FAISS and SentenceTransformer
try:
    import faiss # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # logging.getLogger(__name__).info("FAISS library not found. Semantic search speed might be affected.")

try:
    from sentence_transformers import SentenceTransformer # type: ignore
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    # logging.getLogger(__name__).error("SentenceTransformer library not found! Embedding generation will fail.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING) # type: ignore
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".kg_embeddings_cache"

class KnowledgeGraph:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password",
                 cache_dir: str = DEFAULT_CACHE_DIR): # Added cache_dir parameter
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
            keep_alive=True
        )
        self.uri_identifier = hashlib.md5(uri.encode()).hexdigest()[:8] # Simple ID for the URI for caching
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {uri}")
            self.create_indexes()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j or create indexes: {e}", exc_info=True)
            raise

        # Embedding-related attributes
        self.model: Optional[SentenceTransformer] = None
        self.model_name: Optional[str] = None # Will be set by initialize_embeddings
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"KnowledgeGraph: Embeddings will use device: {self.device}")

        self.entity_embeddings: Optional[torch.Tensor] = None
        self.relation_embeddings: Optional[torch.Tensor] = None
        self.entity_2_id: Optional[Dict[str, int]] = None
        self.id_2_entity: Optional[Dict[int, str]] = None
        self.relation_2_id: Optional[Dict[str, int]] = None
        self.id_2_relation: Optional[Dict[int, str]] = None
        self.entity_faiss_index: Optional[Any] = None

        if FAISS_AVAILABLE:
            logger.info("KnowledgeGraph: FAISS library is available.")
        else:
            logger.warning("KnowledgeGraph: FAISS library not installed. FAISS indexing will be skipped.")
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            logger.error("KnowledgeGraph: SentenceTransformer library not found. Embedding features will be unavailable.")
            # Consider raising an error if this is critical for all use cases of the class

    def _get_cache_filenames(self, model_name_for_cache: str) -> Dict[str, str]:
        """Generates standardized filenames for cache files based on model name and URI."""
        safe_model_name = model_name_for_cache.replace('/', '_').replace('\\', '_')
        prefix = f"{self.uri_identifier}_{safe_model_name}"
        return {
            "entity_embeddings": os.path.join(self.cache_dir, f"{prefix}_entity_embeddings.pt"),
            "entity_map_e2i": os.path.join(self.cache_dir, f"{prefix}_entity_e2i.pkl"),
            "entity_map_i2e": os.path.join(self.cache_dir, f"{prefix}_entity_i2e.pkl"),
            "relation_embeddings": os.path.join(self.cache_dir, f"{prefix}_relation_embeddings.pt"),
            "relation_map_r2i": os.path.join(self.cache_dir, f"{prefix}_relation_r2i.pkl"),
            "relation_map_i2r": os.path.join(self.cache_dir, f"{prefix}_relation_i2r.pkl"),
            "faiss_index": os.path.join(self.cache_dir, f"{prefix}_entity_faiss.index"),
            "metadata": os.path.join(self.cache_dir, f"{prefix}_metadata.json"), # To store counts for validation
        }

    def _save_embeddings_to_cache(self, model_name_for_cache: str) -> None:
        cache_files = self._get_cache_filenames(model_name_for_cache)
        logger.info(f"Saving embeddings and mappings to cache for model '{model_name_for_cache}'...")
        try:
            if self.entity_embeddings is not None: torch.save(self.entity_embeddings, cache_files["entity_embeddings"])
            if self.entity_2_id is not None: 
                with open(cache_files["entity_map_e2i"], 'wb') as f: pickle.dump(self.entity_2_id, f)
            if self.id_2_entity is not None:
                with open(cache_files["entity_map_i2e"], 'wb') as f: pickle.dump(self.id_2_entity, f)
            
            if self.relation_embeddings is not None: torch.save(self.relation_embeddings, cache_files["relation_embeddings"])
            if self.relation_2_id is not None:
                with open(cache_files["relation_map_r2i"], 'wb') as f: pickle.dump(self.relation_2_id, f)
            if self.id_2_relation is not None:
                with open(cache_files["relation_map_i2r"], 'wb') as f: pickle.dump(self.id_2_relation, f)

            if FAISS_AVAILABLE and self.entity_faiss_index is not None:
                faiss.write_index(self.entity_faiss_index, cache_files["faiss_index"])
            
            # Save metadata (e.g., number of entities/relations for basic cache validation)
            metadata = {
                "num_entities_in_db": len(self.get_all_entities()), # Re-query to get current DB state
                "num_relations_in_db": len(self.get_all_relations()),
                "num_entities_embedded": len(self.id_2_entity) if self.id_2_entity else 0,
                "num_relations_embedded": len(self.id_2_relation) if self.id_2_relation else 0,
            }
            with open(cache_files["metadata"], 'w') as f: json.dump(metadata, f)
            logger.info("Successfully saved embeddings and mappings to cache.")
        except Exception as e:
            logger.error(f"Error saving embeddings to cache: {e}", exc_info=True)


    def _load_embeddings_from_cache(self, model_name_for_cache: str) -> bool:
        cache_files = self._get_cache_filenames(model_name_for_cache)
        # Check if all essential cache files exist
        required_files = [
            cache_files["entity_embeddings"], cache_files["entity_map_e2i"], cache_files["entity_map_i2e"],
            cache_files["relation_embeddings"], cache_files["relation_map_r2i"], cache_files["relation_map_i2r"],
            cache_files["metadata"]
        ]
        if FAISS_AVAILABLE: # FAISS index is optional if FAISS was not available during save
             # Check if FAISS index file exists only if we expect it (i.e., embeddings were generated)
            pass # FAISS index loading handled below more carefully

        all_present = all(os.path.exists(f) for f in required_files)
        if not all_present:
            logger.info(f"Cache miss for model '{model_name_for_cache}': Not all cache files found.")
            return False

        logger.info(f"Attempting to load embeddings and mappings from cache for model '{model_name_for_cache}'...")
        try:
            # Basic validation with metadata (optional but good)
            with open(cache_files["metadata"], 'r') as f: cached_metadata = json.load(f)
            
            # Example validation: if current DB has drastically different number of entities/relations,
            # cache might be stale. This is a simple check. More robust checks could involve hashing.
            # current_entities_count = len(self.get_all_entities())
            # if cached_metadata.get("num_entities_in_db") != current_entities_count:
            #     logger.warning(f"Cache for '{model_name_for_cache}' might be stale due to DB entity count change. "
            #                    f"Cached: {cached_metadata.get('num_entities_in_db')}, Current: {current_entities_count}. "
            #                    "Consider recomputing if KG changed significantly.")
            # For simplicity, we'll proceed with loading if files exist and rely on force_recompute.

            self.entity_embeddings = torch.load(cache_files["entity_embeddings"], map_location=self.device)
            with open(cache_files["entity_map_e2i"], 'rb') as f: self.entity_2_id = pickle.load(f)
            with open(cache_files["entity_map_i2e"], 'rb') as f: self.id_2_entity = pickle.load(f)
            
            self.relation_embeddings = torch.load(cache_files["relation_embeddings"], map_location=self.device)
            with open(cache_files["relation_map_r2i"], 'rb') as f: self.relation_2_id = pickle.load(f)
            with open(cache_files["relation_map_i2r"], 'rb') as f: self.id_2_relation = pickle.load(f)

            if FAISS_AVAILABLE and os.path.exists(cache_files["faiss_index"]):
                self.entity_faiss_index = faiss.read_index(cache_files["faiss_index"])
                logger.info("FAISS index loaded from cache.")
            elif FAISS_AVAILABLE and not os.path.exists(cache_files["faiss_index"]) and \
                 self.entity_embeddings is not None and self.entity_embeddings.numel() > 0:
                 logger.warning("FAISS index cache file not found, but entity embeddings exist. FAISS index will be missing for this session unless recomputed.")


            # Ensure model object is also loaded if we successfully load embeddings
            # The model itself is small compared to embeddings, so reloading it is fine.
            # Or, ensure the model_name matches.
            self._load_sentence_transformer_model(model_name_for_cache)
            self.model_name = model_name_for_cache # Crucial to set this

            logger.info("Successfully loaded embeddings and mappings from cache.")
            return True
        except Exception as e:
            logger.error(f"Error loading embeddings from cache, will recompute: {e}", exc_info=True)
            # Clean up potentially partially loaded attributes
            self.entity_embeddings, self.relation_embeddings = None, None
            self.entity_2_id, self.id_2_entity = {}, {}
            self.relation_2_id, self.id_2_relation = {}, {}
            self.entity_faiss_index = None
            return False

    def _load_sentence_transformer_model(self, model_name_arg: str) -> None:
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            logger.error("Cannot load SentenceTransformer: library not available.")
            raise ImportError("SentenceTransformer library is required.")

        if self.model is not None and self.model_name == model_name_arg:
            logger.debug(f"SentenceTransformer model '{model_name_arg}' already loaded.")
            return

        logger.info(f"Loading SentenceTransformer model: '{model_name_arg}' onto device '{self.device}'...")
        try:
            self.model = SentenceTransformer(model_name_arg, device=self.device)
            # self.model_name = model_name_arg # This will be set by the calling function (initialize_embeddings)
                                            # after successful loading or cache hit.
            logger.info(f"Successfully loaded model '{model_name_arg}' for KG instance.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name_arg}': {e}", exc_info=True)
            self.model = None # Ensure model is None on failure
            # self.model_name = None
            raise

    def initialize_embeddings(self, model_name: str, 
                              embedding_encode_batch_size: int = 1024,
                              force_recompute: bool = False):
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            logger.error("Cannot initialize embeddings: SentenceTransformer library not available.")
            return

        self.model_name = model_name # Set model_name first for caching logic

        if not force_recompute:
            if self._load_embeddings_from_cache(model_name):
                # Ensure the SBERT model object is also loaded, _load_embeddings_from_cache calls _load_sentence_transformer_model
                logger.info(f"Embeddings for model '{model_name}' successfully loaded from cache.")
                return # Successfully loaded from cache
            else:
                logger.info(f"Could not load embeddings from cache for '{model_name}', or cache incomplete. Will recompute.")
        else:
            logger.info(f"Forcing recomputation of embeddings for model '{model_name}'.")

        logger.info(f"Initializing in-memory embeddings for all DB items using model='{model_name}'.")
        try:
            self._load_sentence_transformer_model(model_name) # Ensures self.model is loaded
        except Exception as e:
            logger.error(f"Cannot initialize embeddings due to model loading failure for '{model_name}': {e}")
            return # Exit if model cannot be loaded

        self._compute_embeddings_in_memory(embedding_encode_batch_size)
        
        # After computing, save to cache (if successful and embeddings were generated)
        if self.entity_embeddings is not None or self.relation_embeddings is not None: # Check if something was actually computed
            self._save_embeddings_to_cache(model_name)
        else:
            logger.warning("Embeddings computation resulted in no embeddings. Cache not saved.")
            
        logger.info("In-memory embeddings computation and FAISS index (if applicable) complete.")


    def _compute_embeddings_in_memory(self, embedding_encode_batch_size: int):
        if self.model is None:
            logger.critical("CRITICAL: _compute_embeddings_in_memory called but self.model is None. This indicates a logic error in initialization flow.")
            # Ensure attributes are in a consistent state if model isn't loaded
            self.entity_embeddings, self.entity_2_id, self.id_2_entity, self.entity_faiss_index = None, {}, {}, None
            self.relation_embeddings, self.relation_2_id, self.id_2_relation = None, {}, {}
            return

        logger.info("Fetching all unique entity IDs from Neo4j for embedding...")
        entity_ids_to_process = self.get_all_entities()
        logger.info("Fetching all unique relation types from Neo4j for embedding...")
        relation_types_to_process = self.get_all_relations()

        # Default assumption: embeddings are not successfully generated until proven otherwise
        entity_embeddings_successfully_generated = False
        relation_embeddings_successfully_generated = False

        # Process Entities
        if entity_ids_to_process:
            unique_entity_ids = sorted(list(set(eid for eid in entity_ids_to_process if eid and eid.strip())))
            if unique_entity_ids:
                logger.info(f"Generating embeddings for {len(unique_entity_ids)} unique entities...")
                # self.model is guaranteed to be not None here due to the check at the start of the method
                self.entity_embeddings = self.model.encode( # type: ignore
                    unique_entity_ids, batch_size=embedding_encode_batch_size, 
                    convert_to_tensor=True, show_progress_bar=True, device=self.device
                )
                if self.entity_embeddings is not None and self.entity_embeddings.numel() > 0:
                    self.entity_2_id = {ent: i for i, ent in enumerate(unique_entity_ids)}
                    self.id_2_entity = {i: ent for i, ent in enumerate(unique_entity_ids)}
                    entity_embeddings_successfully_generated = True

                    if FAISS_AVAILABLE: # Build FAISS only if embeddings were successful
                        logger.info("Building FAISS index for entity embeddings...")
                        try:
                            cpu_embeddings = self.entity_embeddings.cpu().numpy()
                            dimension = cpu_embeddings.shape[1]
                            # Ensure index is re-initialized if we are recomputing
                            self.entity_faiss_index = faiss.IndexFlatIP(dimension) 
                            self.entity_faiss_index.add(cpu_embeddings)
                            logger.info(f"In-memory FAISS index built for entities (ntotal={self.entity_faiss_index.ntotal}).")
                        except Exception as e:
                            logger.error(f"Failed to build FAISS index for entities: {e}", exc_info=True)
                            self.entity_faiss_index = None # Explicitly None on failure
                else:
                    logger.warning("Entity embedding encoding resulted in an empty or None tensor.")
            else: 
                logger.warning("No valid non-empty entity IDs after filtering for embedding.")
        else: 
            logger.info("No entity IDs found in DB for embedding.")
        
        # If entity embeddings were not successfully generated, reset related attributes
        if not entity_embeddings_successfully_generated:
            logger.info("Resetting entity embedding attributes as no valid embeddings were generated/assigned.")
            self.entity_embeddings, self.entity_2_id, self.id_2_entity, self.entity_faiss_index = None, {}, {}, None

        # Process Relations
        if relation_types_to_process:
            unique_relation_types = sorted(list(set(rel for rel in relation_types_to_process if rel and rel.strip())))
            if unique_relation_types:
                logger.info(f"Generating embeddings for {len(unique_relation_types)} unique relations...")
                self.relation_embeddings = self.model.encode( # type: ignore
                    unique_relation_types, batch_size=embedding_encode_batch_size, 
                    convert_to_tensor=True, show_progress_bar=True, device=self.device
                )
                if self.relation_embeddings is not None and self.relation_embeddings.numel() > 0:
                    self.relation_2_id = {rel: i for i, rel in enumerate(unique_relation_types)}
                    self.id_2_relation = {i: rel for i, rel in enumerate(unique_relation_types)}
                    relation_embeddings_successfully_generated = True
                else:
                    logger.warning("Relation embedding encoding resulted in an empty or None tensor.")
            else: 
                logger.warning("No valid non-empty relation types after filtering for embedding.")
        else: 
            logger.info("No relation types found in DB for embedding.")

        # If relation embeddings were not successfully generated, reset related attributes
        if not relation_embeddings_successfully_generated:
            logger.info("Resetting relation embedding attributes as no valid embeddings were generated/assigned.")
            self.relation_embeddings, self.relation_2_id, self.id_2_relation = None, {}, {}
    def _ensure_model_and_embeddings_initialized(self, # (与您上一版相同)
                                                 check_entity_embeddings: bool = False, 
                                                 check_relation_embeddings: bool = False, 
                                                 check_faiss_index: bool = False) -> bool:
        if not self.model or not self.model_name: # Check if model object and name are set
            logger.error("SentenceTransformer model not loaded or model_name not set. Call initialize_embeddings() first.")
            return False
        if check_entity_embeddings and (self.entity_embeddings is None or self.entity_2_id is None or self.id_2_entity is None):
            logger.error("Entity embeddings/mappings not initialized. Call initialize_embeddings().")
            return False
        if check_relation_embeddings and (self.relation_embeddings is None or self.relation_2_id is None or self.id_2_relation is None):
            logger.error("Relation embeddings/mappings not initialized. Call initialize_embeddings().")
            return False
        if check_faiss_index and FAISS_AVAILABLE and self.entity_faiss_index is None:
            if self.entity_embeddings is not None and self.entity_embeddings.numel() > 0: 
                logger.error("FAISS index for entities not initialized, though embeddings exist. Check initialize_embeddings().")
                return False
        return True
        
    # --- 其他方法 (如 _process_sample, create_indexes, Neo4j 操作, 图查询方法等) ---
    # --- 请确保它们与您在 Turn 13 提供的 KnowledgeGraph 版本中的其余部分保持一致 ---
    # --- 我将从您 Turn 13 的代码中复制这些非词向量核心方法，以确保完整性 ---

    def _process_sample(self, sample: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        try:
            graph_data = sample.get('graph')
            if not graph_data or not isinstance(graph_data, list): return []
            valid_triples: List[Tuple[str, str, str]] = []
            for triple in graph_data:
                if isinstance(triple, (list, tuple)) and len(triple) == 3 and \
                   all(isinstance(item, str) and item.strip() for item in triple): 
                    valid_triples.append(tuple(s.strip() for s in triple)) 
                else: logger.warning(f"Skipping malformed or empty component triple: {triple} in sample.")
            return valid_triples
        except Exception as e:
            logger.error(f"Error processing sample data: {sample}. Error: {e}", exc_info=True)
            return []

    def create_indexes(self):
        try:
            with self.driver.session() as session:
                session.run("CREATE INDEX entity_id_index IF NOT EXISTS FOR (n:ENTITY) ON (n.id)")
                session.run("CREATE INDEX relation_type_index IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)")
                logger.info("Database indexes ensured.")
        except Exception as e: logger.error(f"Failed to create or ensure indexes: {e}", exc_info=True)

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
                else: logger.info("Database is already empty or no nodes found.")
            except Exception as e: logger.error(f"Failed to clear database: {e}", exc_info=True)

    def _batch_process_triples_to_neo4j(self, session, triples: List[Tuple[str, str, str]], batch_size: int = 500):
        def _run_batch_tx(tx, batch_data: List[Tuple[str, str, str]]):
            query = """
            UNWIND $batch as triple
            MERGE (head:ENTITY {id: triple[0]})
            MERGE (tail:ENTITY {id: triple[2]})
            MERGE (head)-[r:RELATION {type: triple[1]}]->(tail)
            """
            tx.run(query, batch=batch_data)
        if not triples:
            logger.info("No triples to load into Neo4j.")
            return
        num_batches = math.ceil(len(triples) / batch_size)
        with tqdm(total=len(triples), desc="Loading triples to Neo4j", unit=" triples") as pbar_triples:
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i + batch_size]
                if batch:
                    try:
                        session.execute_write(_run_batch_tx, batch_data=batch)
                        pbar_triples.update(len(batch)) 
                        pbar_triples.set_postfix_str(f"Batch {i//batch_size + 1}/{num_batches}")
                    except Exception as e: logger.error(f"Error processing batch of triples (starting at index {i}): {e}", exc_info=True)

    def process_cvt_nodes(self):
        # (保持您在 Turn 13 中提供的此方法实现)
        logger.info("Starting CVT node processing...")
        with tqdm(total=2, desc="Processing CVT nodes", unit="step") as pbar_cvt:
            with self.driver.session() as session:
                pbar_cvt.set_description_str("CVT: Creating new relationships (1/2)")
                logger.info("CVT processing: Stage 1/2 - Creating new relationships bypassing CVT nodes...")
                create_new_rels_query = """
                MATCH (h:ENTITY)-[r1:RELATION]->(cvt:ENTITY)-[r2:RELATION]->(t:ENTITY)
                WHERE (cvt.id STARTS WITH 'm.' OR cvt.id STARTS WITH 'g.')
                  AND NOT (h.id STARTS WITH 'm.' OR h.id STARTS WITH 'g.')
                  AND NOT (t.id STARTS WITH 'm.' OR t.id STARTS WITH 'g.')
                WITH h, r1, cvt, r2, t
                MERGE (h)-[new_r:RELATION {type: r1.type + '-' + r2.type}]->(t)
                ON CREATE SET new_r.created_by_cvt_processing = true, 
                              new_r.original_cvt_path = [h.id, r1.type, cvt.id, r2.type, t.id]
                RETURN count(new_r) as created_rels_count_this_batch
                """
                try:
                    result = session.run(create_new_rels_query)
                    summary = result.consume()
                    created_rels_total = summary.counters.relationships_created
                    logger.info(f"CVT processing: Stage 1/2 complete. Relationships created: {created_rels_total}.")
                    pbar_cvt.update(1)
                except Exception as e:
                    logger.error(f"Error during CVT processing (Stage 1/2): {e}", exc_info=True)
                    pbar_cvt.set_description_str("CVT: Error step 1") # type: ignore
                    return
                
                pbar_cvt.set_description_str("CVT: Deleting CVT nodes (2/2)")
                logger.info("CVT processing: Stage 2/2 - Deleting CVT nodes...")
                delete_cvt_query = """
                MATCH (cvt:ENTITY)
                WHERE cvt.id STARTS WITH 'm.' OR cvt.id STARTS WITH 'g.'
                DETACH DELETE cvt
                RETURN count(cvt) as cvts_deleted_count
                """
                try:
                    result = session.run(delete_cvt_query)
                    record = result.single()
                    cvts_deleted_count = record["cvts_deleted_count"] if record and "cvts_deleted_count" in record else 0
                    logger.info(f"CVT processing: Stage 2/2 complete. CVT nodes deleted: {cvts_deleted_count}.")
                    pbar_cvt.update(1)
                except Exception as e:
                    logger.error(f"Error during CVT processing (Stage 2/2): {e}", exc_info=True)
                    pbar_cvt.set_description_str("CVT: Error step 2") # type: ignore
        logger.info("CVT node processing finished.")


    def delete_self_reflexive_edges(self):
        # (保持您在 Turn 13 中提供的此方法实现)
        logger.info("Attempting to delete self-reflexive edges...")
        with tqdm(total=1, desc="Deleting self-reflexive edges", unit="op") as pbar_reflex:
            with self.driver.session() as session:
                delete_query = "MATCH (n:ENTITY)-[r:RELATION]->(n) DELETE r"
                try:
                    result = session.run(delete_query)
                    summary = result.consume() 
                    deleted_count = summary.counters.relationships_deleted
                    logger.info(f"Successfully deleted {deleted_count} self-reflexive_edges.")
                    pbar_reflex.set_postfix_str(f"{deleted_count} deleted", refresh=True) # type: ignore
                except Exception as e:
                    logger.error(f"Error deleting self-reflexive edges: {e}", exc_info=True)
                    pbar_reflex.set_description_str("Deleting self-reflexive (Error!)") # type: ignore
                finally: 
                    if pbar_reflex.n < pbar_reflex.total: # type: ignore
                         pbar_reflex.update(pbar_reflex.total - pbar_reflex.n) # type: ignore
        logger.info("Finished deleting self-reflexive edges.")

    def load_graph_from_dataset(self, 
                                input_source: str, 
                                batch_size_neo4j: int = 500, 
                                hf_dataset_split: Optional[str] = None
                                ):
        # (保持您在 Turn 13 中提供的此方法实现 - 它不直接调用 initialize_embeddings)
        logger.info(f"Loading dataset from source '{input_source}' for KG structure only.")
        dataset_iterable: List[Dict[str, Any]] = []
        if os.path.exists(input_source): 
            if input_source.endswith(".jsonl"):
                try: 
                    with open(input_source, 'r', encoding='utf-8') as f:
                        dataset_iterable = [json.loads(line) for line in f]
                except Exception as e: logger.error(f"Failed to load JSONL: {e}", exc_info=True); return
            elif input_source.endswith(".json"):
                try:
                    with open(input_source, 'r', encoding='utf-8') as f: dataset_iterable = json.load(f)
                    if not isinstance(dataset_iterable, list): logger.error(f"JSON file not a list."); return
                except Exception as e: logger.error(f"Failed to load JSON: {e}", exc_info=True); return
            else: 
                try:
                    from datasets import load_from_disk # type: ignore
                    dataset_iterable = list(load_from_disk(input_source)) # type: ignore
                except Exception as e: logger.error(f"Failed to load from disk: {e}", exc_info=True); return
        else: 
            if not hf_dataset_split: logger.error(f"'{input_source}' not local, 'hf_dataset_split' required."); return
            from datasets import load_dataset as hf_load_dataset # type: ignore
            try:
                logger.info(f"Loading from Hub: '{input_source}', split: '{hf_dataset_split}'")
                dataset_iterable = list(hf_load_dataset(input_source, split=hf_dataset_split)) # type: ignore
                logger.info("Conversion from Hub to list complete.")
            except Exception as e: logger.error(f"Failed to load from Hub: {e}", exc_info=True); return

        if not dataset_iterable: logger.warning(f"Dataset empty from '{input_source}'."); return
        logger.info(f"Dataset loaded: {len(dataset_iterable)} samples for triples.")
        
        all_triples_to_load: List[Tuple[str, str, str]] = []
        for sample in tqdm(dataset_iterable, desc="Extracting triples", unit="samples"):
            triples = self._process_sample(sample)
            if triples: all_triples_to_load.extend(triples)
        logger.info(f"Extracted {len(all_triples_to_load)} raw triples.")

        if all_triples_to_load:
            unique_triples_to_load: List[Tuple[str, str, str]] = []
            with tqdm(total=3, desc="Deduplicating & Sorting Triples", unit="step") as pbar_dedup:
                try:
                    pbar_dedup.set_description_str("Deduplicating (set operation)") # type: ignore
                    unique_set = set(all_triples_to_load)
                    pbar_dedup.update(1)
                    pbar_dedup.set_description_str("Converting set to list") # type: ignore
                    list_of_unique_triples = list(unique_set)
                    pbar_dedup.update(1)
                    pbar_dedup.set_description_str("Sorting unique triples") # type: ignore
                    unique_triples_to_load = sorted(list_of_unique_triples)
                    pbar_dedup.update(1)
                    pbar_dedup.set_postfix_str(f"{len(unique_triples_to_load)} unique", refresh=True) # type: ignore
                    logger.info(f"Deduplicated to {len(unique_triples_to_load)} unique triples.")
                except MemoryError: logger.error("MemoryError during dedup/sort!", exc_info=True); raise 
                except Exception as e: logger.error(f"Error during dedup/sort: {e}", exc_info=True); raise
            
            logger.info(f"Loading {len(unique_triples_to_load)} unique triples into Neo4j...")
            with self.driver.session() as session:
                self._batch_process_triples_to_neo4j(session, unique_triples_to_load, batch_size_neo4j)
            logger.info("Finished loading triples into Neo4j.")
        else: logger.warning("No triples to load.")
        self.process_cvt_nodes()
        self.delete_self_reflexive_edges()
        try: 
            with self.driver.session() as session:
                stats = session.run("MATCH (n:ENTITY) WITH count(n) as ec MATCH ()-[r:RELATION]->() WITH ec, count(r) as rc RETURN ec, rc").single()
                if stats: logger.info(f"DB stats: Entities: {stats['ec']}, Relationships: {stats['rc']}")
        except Exception as e: logger.warning(f"Could not get DB stats: {e}")


    # --- Semantic Search Methods (requiring initialized embeddings) ---
    def get_related_entities_by_question(self, question: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self._ensure_model_and_embeddings_initialized(check_entity_embeddings=True, check_faiss_index=FAISS_AVAILABLE): return []
        # ... (保持您在 Turn 12 中提供的此方法实现，确保 self.model, self.entity_embeddings, self.id_2_entity, self.entity_faiss_index 正常工作)
        # Simplified version for brevity here:
        if self.model is None or self.entity_embeddings is None or self.id_2_entity is None: return []
        q_emb = self.model.encode(question, convert_to_tensor=True, device=self.device)
        if FAISS_AVAILABLE and self.entity_faiss_index:
            q_emb_np = q_emb.cpu().numpy().reshape(1,-1)
            D, I = self.entity_faiss_index.search(q_emb_np, top_k)
            return [(self.id_2_entity[i], float(d)) for i, d in zip(I[0], D[0]) if i != -1 and i in self.id_2_entity]
        else: # Fallback
            sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), self.entity_embeddings)
            scores, indices = torch.topk(sims, k=min(top_k, len(sims)))
            return [(self.id_2_entity[idx.item()], score.item()) for score, idx in zip(scores, indices) if idx.item() in self.id_2_entity]


    def get_related_relations_by_question(self, entity_id: str, question: str, top_k: int = 5, direction: str = "out") -> List[Tuple[str, float]]:
        if not self._ensure_model_and_embeddings_initialized(check_relation_embeddings=True): return []
        # ... (保持您在 Turn 12 中提供的此方法实现，确保 self.model, self.relation_embeddings, self.id_2_relation, self.relation_2_id 正常工作)
        # Simplified version for brevity here:
        if self.model is None or self.relation_embeddings is None or self.id_2_relation is None or self.relation_2_id is None : return []
        q_emb = self.model.encode(question, convert_to_tensor=True, device=self.device)
        
        candidate_rels_db = self.get_related_relations(entity_id, direction=direction)
        if not candidate_rels_db: return []

        valid_rels_for_scoring = [r for r in candidate_rels_db if r in self.relation_2_id]
        if not valid_rels_for_scoring: return []
        
        rel_indices = [self.relation_2_id[r] for r in valid_rels_for_scoring]
        rel_embs_subset = self.relation_embeddings[torch.tensor(rel_indices, device=self.relation_embeddings.device)]

        if rel_embs_subset.numel() == 0: return []
        sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), rel_embs_subset)
        scores, indices = torch.topk(sims, k=min(top_k, len(sims)))
        return [(valid_rels_for_scoring[idx.item()], score.item()) for score, idx in zip(scores, indices)]


    # --- Standard Graph Query Methods (保持您在 Turn 13 中提供的这些方法实现) ---
    def get_shortest_paths(self, source_id: str, target_id: str, max_depth: int = 5) -> List[List[Tuple[str, str, str]]]:
        paths_result: List[List[Tuple[str, str, str]]] = []
        if not source_id or not target_id: return paths_result 
        if max_depth <= 0: 
            if source_id == target_id: logger.debug("Shortest path query: src=tgt, max_depth<=0.")
            return paths_result 
        if source_id == target_id: return paths_result
        query = f"MATCH (s:ENTITY {{id: $source_id}}), (t:ENTITY {{id: $target_id}}) MATCH p = allShortestPaths((s)-[:RELATION*1..{max_depth}]->(t)) RETURN p"
        try:
            with self.driver.session() as session: 
                results = session.run(query, source_id=source_id, target_id=target_id)
                for record in results:
                    path_obj = record["p"]
                    path_tuples: List[Tuple[str, str, str]] = []
                    if not path_obj.nodes or not path_obj.relationships: continue
                    valid_path = True
                    for i, rel_obj in enumerate(path_obj.relationships):
                        start_node, end_node = path_obj.nodes[i], path_obj.nodes[i+1]
                        h, r, t = start_node.get('id'), rel_obj.get('type'), end_node.get('id')
                        if not all([h, r, t]): valid_path = False; break
                        path_tuples.append((h, r, t))
                    if valid_path and path_tuples: paths_result.append(path_tuples)
        except Exception as e: logger.error(f"Error in get_shortest_paths: {e}", exc_info=True)
        return paths_result

    def get_target_entities(self, source_id: str, relation_type: str, direction: str = "out") -> List[str]:
        if direction not in ["in", "out"]: return []
        q_template = ("MATCH (:ENTITY {id: $sid})-[r:RELATION {type: $rtype}]->(t:ENTITY) RETURN DISTINCT t.id as tid" if direction == "out"
                      else "MATCH (t:ENTITY)-[r:RELATION {type: $rtype}]->(:ENTITY {id: $sid}) RETURN DISTINCT t.id as tid")
        try:
            with self.driver.session() as session:
                res = session.run(q_template, sid=source_id, rtype=relation_type)
                return [r["tid"] for r in res if r["tid"]]
        except Exception as e: logger.error(f"Error in get_target_entities: {e}", exc_info=True); return []
            
    def get_related_relations(self, entity_id: str, direction: str = "out") -> List[str]:
        if direction not in ["in", "out"]: return []
        q_template = ("MATCH (:ENTITY {id: $eid})-[r:RELATION]->() WHERE r.type IS NOT NULL RETURN DISTINCT r.type as rtype" if direction == "out"
                      else "MATCH ()-[r:RELATION]->(:ENTITY {id: $eid}) WHERE r.type IS NOT NULL RETURN DISTINCT r.type as rtype")
        try:
            with self.driver.session() as session:
                res = session.run(q_template, eid=entity_id)
                return [r["rtype"] for r in res if r["rtype"]]
        except Exception as e: logger.error(f"Error in get_related_relations: {e}", exc_info=True); return []

    def get_all_entities(self) -> List[str]:
        with self.driver.session() as session: 
            q = "MATCH (n:ENTITY) WHERE n.id IS NOT NULL RETURN DISTINCT n.id AS entity_id"
            try: return [r["entity_id"] for r in session.run(q) if r["entity_id"]]
            except Exception as e: logger.error(f"Error getting all entities: {e}", exc_info=True); return []

    def get_all_relations(self) -> List[str]:
        with self.driver.session() as session: 
            q = "MATCH ()-[r:RELATION]->() WHERE r.type IS NOT NULL RETURN DISTINCT r.type AS relation_type"
            try: return [r["relation_type"] for r in session.run(q) if r["relation_type"]]
            except Exception as e: logger.error(f"Error getting all relations: {e}", exc_info=True); return []