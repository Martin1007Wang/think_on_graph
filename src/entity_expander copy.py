# Necessary imports (ensure these align with your project structure)
import logging
import threading
import time
from typing import Optional, List, Dict, Any, Tuple, Set, ClassVar, FrozenSet
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict # Use defaultdict for easier grouping

# Assuming these imports are correctly set up in your project
try:
    from src.knowledge_graph import KnowledgeGraph
    from src.model_interface import ModelInterface # Assumes uses LLMOutputParserOptimized
    from src.path_manager import PathManager
    from src.utils.data_utils import EntityExpansion, EntityRelation
except ImportError:
    # Fallback for standalone execution/testing - replace with dummy classes if needed
    class KnowledgeGraph:
        # Dummy methods for testing the grouping logic
        _graph_data = {
            "CVT_1": {"attr1": ["targetA"], "attr2": ["targetB"]},
            "CVT_2": {"attr2": ["targetB"], "attr1": ["targetA"]}, # Same structure as CVT_1
            "CVT_3": {"attr1": ["targetA"], "attr3": ["targetC"]}, # Different structure
            "CVT_4": {"attr1": ["targetA"], "attr2": ["targetB"]}, # Same structure as CVT_1
            "CVT_5": {}, # Empty CVT
        }
        def get_related_relations(self, node: str, direction: str) -> List[str]:
            if direction == "out" and node in self._graph_data:
                return list(self._graph_data[node].keys())
            return []
        def get_target_entities(self, node: str, relation: str, direction: str) -> List[str]:
             if direction == "out" and node in self._graph_data and relation in self._graph_data[node]:
                 return self._graph_data[node][relation]
             return []

    class ModelInterface: pass
    class PathManager:
        def is_coded_entity(self, entity: str) -> bool: return entity.startswith("CVT_") # Example dummy logic
    class EntityExpansion:
        def __init__(self, entity): self.entity = entity; self.relations = []
    class EntityRelation:
        def __init__(self, relation, targets, metadata=None): self.relation = relation; self.targets = targets; self.metadata = metadata if metadata else {}

logger = logging.getLogger(__name__)
# Basic config if not set externally
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')


class EntityExpander:
    RELATION_SELECTION_TEMPLATE: ClassVar[str] = "relation_selection"
    RELATION_SELECTION_WITH_HISTORY_TEMPLATE: ClassVar[str] = "relation_selection_history"
    CVT_ATTRIBUTE_SELECTION_TEMPLATE: ClassVar[str] = "cvt_attribute_selection" # Template name used by predict_model
    CVT_ATTRIBUTE_PARSER_METHOD_NAME: ClassVar[str] = "parse_attribute_selection" # Method predict_model's parser MUST have

    def __init__(self,
                 kg: KnowledgeGraph,
                 explore_model_interface: ModelInterface, # Used for relation selection
                 predict_model_interface: ModelInterface, # Used for CVT attribute selection
                 path_manager: PathManager,
                 max_workers: int = 8,
                 llm_concurrency_limit: int = 8,
                 max_relation_selection_count: int = 5, # Renamed for clarity
                 min_cvts_for_parallel_processing: int = 3): # Adjusted default slightly, tune as needed

        if llm_concurrency_limit < 1: raise ValueError("llm_concurrency_limit must be >= 1")
        if max_workers < 1: raise ValueError("max_workers must be >= 1")
        # MODIFIED: min_cvts_for_parallel_processing now applies to groups
        if min_cvts_for_parallel_processing < 1: raise ValueError("min_cvts_for_parallel_processing must be >= 1")

        self.kg = kg
        self.explore_model = explore_model_interface
        self.predict_model = predict_model_interface # Correct model for attribute selection
        self.path_manager = path_manager
        self.max_relation_selection_count = max_relation_selection_count # Max relations to explore from an entity
        self.max_workers = max_workers
        self.llm_semaphore = threading.Semaphore(llm_concurrency_limit)
        self.min_cvts_for_parallel_processing = min_cvts_for_parallel_processing

        # --- Initialization Checks --- (Remain the same)
        parser_instance = getattr(self.predict_model, 'parser', None)
        if not parser_instance or not hasattr(parser_instance, self.CVT_ATTRIBUTE_PARSER_METHOD_NAME):
            logger.error(f"CRITICAL: LLMOutputParser used by predict_model_interface "
                         f"does NOT have the required '{self.CVT_ATTRIBUTE_PARSER_METHOD_NAME}' method. "
                         f"CVT attribute selection parsing WILL FAIL.")
            # raise AttributeError(f"Predict model's parser missing required method: {self.CVT_ATTRIBUTE_PARSER_METHOD_NAME}")
        else:
            logger.info(f"CVT attribute selection parser '{self.CVT_ATTRIBUTE_PARSER_METHOD_NAME}' found on predict_model.")

        if not hasattr(self.predict_model, 'select_attributes'):
             logger.warning(f"Predict model interface does not have a 'select_attributes' method. "
                            f"Ensure the method called in _execute_llm_attribute_selection exists.")

        logger.info(f"EntityExpander initialized with max_workers={self.max_workers}, "
                    f"llm_concurrency_limit={llm_concurrency_limit}, "
                    f"max_relation_selection_count={self.max_relation_selection_count}, "
                    f"min_cvts_for_parallel_processing applied to GROUPS={self.min_cvts_for_parallel_processing}") # Log change

    # --- Core Public Method ---
    def expand_entity(self, entity: str, question: str, history: str) -> Optional[EntityExpansion]:
        logger.info(f"Expanding entity: '{entity}' for question: '{question[:50]}...'")
        start_time = time.monotonic()

        # 1. Fetch initial outgoing relations
        all_relations = self._fetch_relations(entity)
        if not all_relations:
            logger.debug(f"No outgoing relations found for entity '{entity}'. Cannot expand.")
            return None

        # 2. Select promising relations using explore_model LLM
        selected_relations: List[str] = []
        try:
            selected_relations = self._execute_llm_relation_selection(
                entity=entity, relations=all_relations, question=question,
                history=history, max_selection_count=self.max_relation_selection_count
            )
        except Exception as e:
            logger.error(f"Error during relation selection task for entity '{entity}': {e}", exc_info=True)

        if not selected_relations:
            logger.info(f"LLM selected no promising relations for entity '{entity}' based on the question. Stopping expansion here.")
            return None

        logger.info(f"LLM selected {len(selected_relations)} relations for '{entity}': {selected_relations}")

        # 3. Process selected relations: Identify direct links and *potential* CVT tasks
        expansion_result = EntityExpansion(entity=entity)
        # MODIFICATION: Store potential tasks before grouping
        potential_cvt_tasks: List[Tuple[str, str, str, str]] = [] # (source_relation, cvt_id, question, history_to_cvt)

        for relation in selected_relations:
            try:
                targets = self.kg.get_target_entities(entity, relation, "out")
                if not targets:
                    logger.debug(f"No targets found for selected relation '{entity}' -> '{relation}'")
                    continue

                named_entities, coded_entities = self._separate_entities(targets)

                # Add direct named entity results immediately
                if named_entities:
                    direct_relation = EntityRelation(relation=relation, targets=named_entities)
                    expansion_result.relations.append(direct_relation)
                    logger.debug(f"Added {len(named_entities)} direct named target(s) for '{entity}' -> '{relation}'")

                # Collect potential CVT tasks for grouping
                if coded_entities:
                    logger.debug(f"Found {len(coded_entities)} potential CVT target(s) for '{entity}' -> '{relation}'. Queuing for grouping.")
                    for cvt_id in coded_entities:
                        # History leading *to* this specific CVT
                        history_to_cvt = f"{history} --> {entity} --[{relation}]--> {cvt_id}"
                        potential_cvt_tasks.append(
                            (relation, cvt_id, question, history_to_cvt)
                        )
            except Exception as e:
                logger.error(f"Error processing relation '{relation}' targets for entity '{entity}': {e}", exc_info=True)
                # Continue to the next relation

        # 3.5 NEW: Group potential CVT tasks by structure
        # grouped_tasks: Dict[Tuple[str, FrozenSet[Tuple[str, FrozenSet[str]]]], List[Tuple[str, str, str]]]
        # Key: (source_relation, frozenset( (out_rel, frozenset(out_targets)) ))
        # Value: List of (cvt_id, question, history_to_cvt) tuples belonging to the group
        grouped_tasks = defaultdict(list)
        # Store fetched structures to avoid refetching inside the loop
        structure_cache: Dict[str, Optional[Dict[str, List[str]]]] = {}

        logger.debug(f"Grouping {len(potential_cvt_tasks)} potential CVT tasks...")
        for source_relation, cvt_id, q, hist_to_cvt in potential_cvt_tasks:
            try:
                if cvt_id in structure_cache:
                     cvt_structure = structure_cache[cvt_id]
                     if cvt_structure is None: continue # Skip if fetching failed previously
                else:
                    cvt_structure = self._fetch_cvt_outgoing_structure(cvt_id)
                    structure_cache[cvt_id] = cvt_structure # Cache result (even if None)

                if not cvt_structure: # Handles None or empty dict {}
                    logger.debug(f"CVT '{cvt_id}' has no outgoing structure or fetch failed. Skipping.")
                    continue

                # Create canonical representation of the structure for the key
                # Sort relations, then sort targets within each relation
                canonical_structure_items = []
                for out_rel, out_targets in sorted(cvt_structure.items()):
                    # Ensure targets are strings and sorted
                    sorted_targets = tuple(sorted([str(t).strip() for t in out_targets if t and str(t).strip()]))
                    if sorted_targets: # Only include relations with valid targets
                      canonical_structure_items.append((out_rel, frozenset(sorted_targets))) # Use frozenset for targets

                if not canonical_structure_items:
                     logger.debug(f"CVT '{cvt_id}' had structure but no valid target strings after cleaning. Skipping.")
                     continue

                # Key: incoming relation + canonical outgoing structure
                group_key = (source_relation, frozenset(canonical_structure_items))

                # Value: List of (cvt_id, question, history_to_cvt)
                grouped_tasks[group_key].append((cvt_id, q, hist_to_cvt))

            except Exception as group_e:
                logger.error(f"Error fetching structure or grouping CVT '{cvt_id}': {group_e}", exc_info=True)

        num_cvt_groups = len(grouped_tasks)
        if num_cvt_groups > 0:
             logger.info(f"Grouped {len(potential_cvt_tasks)} potential tasks into {num_cvt_groups} distinct CVT structure groups.")

             # 4. MODIFIED: Execute CVT Expansion Tasks based on GROUPS
             processed_group_count = 0
             # Apply parallel processing threshold to the number of groups
             use_parallel_group_processing = num_cvt_groups >= self.min_cvts_for_parallel_processing
             tasks_to_process = list(grouped_tasks.items()) # List of (group_key, list_of_task_tuples)

             if use_parallel_group_processing:
                 actual_workers = min(num_cvt_groups, self.max_workers)
                 logger.debug(f"Processing {num_cvt_groups} CVT groups in parallel using {actual_workers} workers.")
                 with ThreadPoolExecutor(max_workers=actual_workers, thread_name_prefix=f'ExpandCVTGroup_{entity[:5]}_') as executor:
                     # Future maps to the group_key for logging/debugging
                     future_to_group_key = {
                         executor.submit(self._expand_cvt_group, entity, group_key, group_tasks): group_key
                         for group_key, group_tasks in tasks_to_process
                     }
                     for future in as_completed(future_to_group_key):
                         group_key = future_to_group_key[future]
                         processed_group_count += 1
                         try:
                             # _expand_cvt_group returns a *single* EntityRelation or None
                             group_expansion_relation: Optional[EntityRelation] = future.result()
                             if group_expansion_relation:
                                 expansion_result.relations.append(group_expansion_relation)
                                 cvt_ids_in_group = [t[0] for t in grouped_tasks[group_key]] # Extract CVT IDs for logging
                                 logger.debug(f"Successfully added results from parallel expansion of CVT group (Key Relation: {group_key[0]}, {len(cvt_ids_in_group)} CVTs: {cvt_ids_in_group[:3]}...).")
                         except Exception as e:
                             logger.error(f"Error processing parallel result for CVT group (Key Relation: {group_key[0]}): {e}", exc_info=True)
             else:
                 # Process sequentially for fewer groups
                 logger.debug(f"Processing {num_cvt_groups} CVT groups sequentially.")
                 for group_key, group_tasks in tasks_to_process:
                     processed_group_count += 1
                     try:
                         group_expansion_relation = self._expand_cvt_group(entity, group_key, group_tasks)
                         if group_expansion_relation:
                             expansion_result.relations.append(group_expansion_relation)
                             cvt_ids_in_group = [t[0] for t in group_tasks] # Extract CVT IDs for logging
                             logger.debug(f"Successfully added results from sequential expansion of CVT group (Key Relation: {group_key[0]}, {len(cvt_ids_in_group)} CVTs: {cvt_ids_in_group[:3]}...).")
                     except Exception as e:
                         logger.error(f"Error during sequential expansion of CVT group (Key Relation: {group_key[0]}): {e}", exc_info=True)

             logger.info(f"Finished processing {processed_group_count}/{num_cvt_groups} CVT groups for entity '{entity}'.")


        # 5. Deduplicate final results (No change needed here, relies on normalized_path)
        final_relation_count = 0
        if expansion_result.relations:
            original_count = len(expansion_result.relations)
            expansion_result = self._deduplicate_results(expansion_result)
            final_relation_count = len(expansion_result.relations)
            if original_count > final_relation_count:
                logger.info(f"Deduplication reduced paths/relations from {original_count} to {final_relation_count} for entity '{entity}'.")

        duration = time.monotonic() - start_time
        logger.info(f"Finished expanding entity '{entity}'. Found {final_relation_count} total unique paths/relations. "
                    f"Duration: {duration:.2f}s")

        return expansion_result if final_relation_count > 0 else None

    # --- NEW Helper to Fetch CVT Structure ---
    def _fetch_cvt_outgoing_structure(self, cvt_id: str) -> Optional[Dict[str, List[str]]]:
        """Fetches the outgoing relations and their targets for a given CVT ID."""
        logger.debug(f"Fetching outgoing structure for CVT '{cvt_id}'")
        details: Dict[str, List[str]] = {}
        try:
            relations = self.kg.get_related_relations(cvt_id, "out")
            if not relations:
                logger.debug(f"No outgoing relations found for CVT '{cvt_id}'.")
                return {} # Return empty dict, not None, if no relations

            for rel in relations:
                rel_key = str(rel).strip()
                if not rel_key: continue # Skip empty relation names

                targets = self.kg.get_target_entities(cvt_id, rel_key, "out")
                if targets:
                    target_list = list(targets) if isinstance(targets, (list, tuple, set)) else [targets]
                    valid_targets = [str(t).strip() for t in target_list if t is not None and str(t).strip()]
                    if valid_targets:
                        details[rel_key] = valid_targets # Store cleaned list
            return details # Return dict, potentially empty if no valid targets found

        except Exception as kg_e:
            logger.error(f"Failed to fetch details for CVT '{cvt_id}': {kg_e}", exc_info=False) # Keep log less verbose on KG errors
            return None # Indicate failure to fetch

    # --- NEW Helper to Expand a Group of CVTs ---
    def _expand_cvt_group(self,
                          source_entity: str,
                          group_key: Tuple[str, FrozenSet[Tuple[str, FrozenSet[str]]]],
                          group_tasks: List[Tuple[str, str, str]]) -> Optional[EntityRelation]:
        """
        Expands a group of CVTs that share the same incoming relation and outgoing structure.
        Performs attribute selection once for the group.

        Args:
            source_entity: The entity from which the expansion originates.
            group_key: The key identifying the group (source_relation, canonical_structure).
            group_tasks: List of (cvt_id, question, history_to_cvt) tuples in the group.

        Returns:
            A single EntityRelation representing the chosen expansion path for the group, or None.
        """
        if not group_tasks:
            return None

        # Extract representative info from the first task in the group
        source_relation = group_key[0]
        representative_task = group_tasks[0]
        representative_cvt_id = representative_task[0]
        question = representative_task[1]
        # Use a generic history or the first task's history? Using first for now.
        # A more robust history might omit the final specific CVT ID.
        history = representative_task[2] # History leading to the first CVT in the group

        list_of_cvt_ids = [task[0] for task in group_tasks]
        logger.debug(f"Expanding CVT group via '{source_relation}' (Rep CVT: {representative_cvt_id}, {len(list_of_cvt_ids)} total CVTs)")
        expansion_start_time = time.monotonic()

        # 1. Format Available Attributes from the canonical structure
        canonical_structure = group_key[1]
        available_attributes: List[str] = []
        seen_attributes: Set[str] = set()
        logger.debug(f"Formatting {len(canonical_structure)} unique relation types for group selection...")

        for relation_str, targets_frozenset in canonical_structure:
            # Targets are already strings and unique within the frozenset
            for target_str in targets_frozenset:
                attribute = f"{relation_str} -> {target_str}"
                # Should be unique already due to frozenset structure, but double-check
                if attribute not in seen_attributes:
                    available_attributes.append(attribute)
                    seen_attributes.add(attribute)

        if not available_attributes:
            logger.error(f"Internal Error: No unique attributes formatted for LLM selection from group (Rep CVT: {representative_cvt_id}), despite having structure key.")
            return None
        logger.debug(f"Prepared {len(available_attributes)} unique Rel->Tgt attributes for selection from CVT group (Rep CVT: {representative_cvt_id}).")

        # 2. Call LLM (predict_model) to Select the Most Relevant Attribute Pair *once* for the group
        selected_attribute_strings: Optional[List[str]] = None
        try:
            # Use the representative CVT ID and history for the LLM call context
            selected_attribute_strings = self._execute_llm_attribute_selection(
                question=question,
                history=history,
                cvt_id=representative_cvt_id, # Pass representative ID for context
                source_entity=source_entity,
                source_relation=source_relation,
                attributes=available_attributes,
                max_selection_count=1 # Still want only the single best path via this structure
            )
        except Exception as llm_e:
            logger.error(f"Error calling LLM attribute selection helper for CVT group (Rep CVT: {representative_cvt_id}): {llm_e}", exc_info=True)
            return None # LLM call failed

        # 3. Process Selection Result
        if not selected_attribute_strings: # Handles None or empty list []
            logger.debug(f"LLM selected no relevant Rel->Tgt attribute from CVT group (Rep CVT: {representative_cvt_id}). Group path terminates.")
            return None

        selected_string = selected_attribute_strings[0]
        logger.debug(f"LLM selected attribute string: '{selected_string}' for CVT group (Rep CVT: {representative_cvt_id})")

        # 4. Parse the selected "relation -> target" string
        try:
            parts = selected_string.split(" -> ", 1)
            if len(parts) == 2:
                selected_relation = parts[0].strip()
                selected_target = parts[1].strip()

                if not selected_relation or not selected_target:
                    logger.warning(f"Parsed empty relation or target from selected string '{selected_string}' for CVT group (Rep CVT: {representative_cvt_id}). Cannot proceed.")
                    return None

                logger.info(f"LLM selected attribute '{selected_relation}' with target '{selected_target}' from CVT group as the relevant path segment.")

                # 5. Construct ONE 2-Hop Path Result for the entire group
                # Use representative CVT ID in the human-readable path? Or maybe indicate group?
                # Using first CVT ID for now for consistency in logging/debugging downstream
                first_cvt_id = list_of_cvt_ids[0]
                two_hop_path_key = f"{source_relation} > {first_cvt_id} [Group:{len(list_of_cvt_ids)}] > {selected_relation}" # Indicate group
                normalized_path_key = f"{source_relation}>{selected_relation}" # Key for deduplication

                # Create the single relation representing the choice for the group
                result_relation = EntityRelation(
                    relation=two_hop_path_key, # Use the descriptive path
                    targets=[selected_target], # The single target selected
                    metadata={
                        "is_cvt_path": True,
                        "is_cvt_group": True, # New flag
                        "merged_cvt_ids": sorted(list_of_cvt_ids), # Store all IDs from the group
                        "normalized_path": normalized_path_key, # Used for deduplication
                        "source_relation": source_relation,
                        "selected_relation": selected_relation,
                        "selected_target": selected_target,
                        "representative_cvt_id": representative_cvt_id # Store which one was used for LLM context
                    }
                )

                duration = time.monotonic() - expansion_start_time
                logger.debug(f"Successfully constructed expansion through CVT group (Rep CVT: {representative_cvt_id}) via attribute '{selected_relation}'. Duration: {duration:.2f}s")
                # Return the single relation, not a full EntityExpansion here
                return result_relation

            else:
                logger.error(f"Could not parse selected attribute string '{selected_string}' into 'relation -> target' format for CVT group. Selection invalid.")
                return None

        except Exception as parse_e:
            logger.error(f"Error parsing or processing LLM attribute selection result ('{selected_string}') for CVT group: {parse_e}", exc_info=True)
            return None


    # --- CVT and Relation Processing Helpers --- (Keep _deduplicate_results, _separate_entities)

    def _deduplicate_results(self, expansion: EntityExpansion) -> EntityExpansion:
        """
        Deduplicates results in the EntityExpansion object.
        - Direct relations (no 'normalized_path' metadata): Keyed by (relation, target)
        - CVT path relations ('normalized_path' metadata): Keyed by (normalized_path, target)
        Merges metadata for deduplicated CVT paths, especially 'merged_cvt_ids'.

        Args:
            expansion: The EntityExpansion object with potentially redundant relations.

        Returns:
            A new EntityExpansion object with deduplicated relations.
        """
        if not expansion or not expansion.relations:
            return expansion

        unique_relations_map: Dict[str, EntityRelation] = {}
        processed_keys: Set[str] = set()

        for relation_obj in expansion.relations:
            is_cvt_path = relation_obj.metadata and "normalized_path" in relation_obj.metadata
            base_relation_name = relation_obj.relation # Original relation name or constructed path

            for target in relation_obj.targets:
                target_str = str(target)
                key: str
                if is_cvt_path:
                    norm_path = relation_obj.metadata["normalized_path"]
                    key = f"CVT:{norm_path}::{target_str}"
                else:
                    key = f"DIRECT:{base_relation_name}::{target_str}"

                if key not in processed_keys:
                    processed_keys.add(key)
                    if len(relation_obj.targets) == 1:
                        unique_relations_map[key] = relation_obj
                    else:
                        new_relation = EntityRelation(
                            relation=base_relation_name,
                            targets=[target],
                            metadata=relation_obj.metadata.copy() if relation_obj.metadata else {}
                        )
                        unique_relations_map[key] = new_relation
                else:
                    # Duplicate found
                    existing_relation = unique_relations_map.get(key)
                    if existing_relation and is_cvt_path:
                        # Merge CVT IDs if it's a CVT path duplicate
                        current_cvt_ids = set(relation_obj.metadata.get("merged_cvt_ids", [relation_obj.metadata.get("cvt_id", "unknown")]))

                        if "merged_cvt_ids" not in existing_relation.metadata:
                             # Initialize from the existing relation's original ID if needed
                             original_cvt_id = existing_relation.metadata.get("cvt_id", "unknown")
                             existing_relation.metadata["merged_cvt_ids"] = {original_cvt_id} # Use a set for easier merging

                        # Ensure merged_ids is a set
                        if not isinstance(existing_relation.metadata["merged_cvt_ids"], set):
                             existing_relation.metadata["merged_cvt_ids"] = set(existing_relation.metadata["merged_cvt_ids"])

                        # Add new IDs
                        initial_count = len(existing_relation.metadata["merged_cvt_ids"])
                        existing_relation.metadata["merged_cvt_ids"].update(current_cvt_ids)
                        if len(existing_relation.metadata["merged_cvt_ids"]) > initial_count:
                            logger.debug(f"Deduplicating: Merged CVT IDs for key '{key}'. New set: {existing_relation.metadata['merged_cvt_ids']}")
                        else:
                            logger.debug(f"Deduplicating: Skipping redundant CVT path for key '{key}' (IDs already present)")

                    else: # Duplicate direct relation or other issue
                        logger.debug(f"Deduplicating: Skipping redundant path for key '{key}'")


        # Convert merged sets back to sorted lists for consistent output
        final_relations = []
        for rel in unique_relations_map.values():
            if isinstance(rel.metadata.get("merged_cvt_ids"), set):
                rel.metadata["merged_cvt_ids"] = sorted(list(rel.metadata["merged_cvt_ids"]))
            final_relations.append(rel)

        deduplicated_expansion = EntityExpansion(entity=expansion.entity)
        deduplicated_expansion.relations = final_relations
        return deduplicated_expansion


    def _separate_entities(self, entities: List[Any]) -> Tuple[List[str], List[str]]:
        """Separates entities into named and coded/CVT entity strings, ensuring uniqueness."""
        # (No changes needed in this method)
        if not entities:
            return [], []
        is_coded_or_cvt = self.path_manager.is_coded_entity
        named_entities_set: Set[str] = set()
        coded_entities_set: Set[str] = set()
        for entity_obj in entities:
            try:
                entity_str = str(entity_obj).strip()
                if not entity_str: continue
                if is_coded_or_cvt(entity_str):
                    coded_entities_set.add(entity_str)
                else:
                    named_entities_set.add(entity_str)
            except Exception as conversion_err:
                logger.warning(f"Error processing or classifying target entity '{entity_obj!r}': {conversion_err}", exc_info=False)
                continue
        return list(named_entities_set), list(coded_entities_set)


    # --- LLM Execution Helpers --- (No changes needed)

    def _execute_llm_relation_selection(self, entity: str, relations: List[str],
                                        question: str, history: str,
                                        max_selection_count: int) -> List[str]:
         # (No change from previous version)
        if not relations: return []
        with self.llm_semaphore:
            logger.debug(f"Acquired semaphore for relation selection LLM call (Entity: {entity}).")
            start_llm_time = time.monotonic()
            try:
                selected = self.explore_model.select_relations(
                    entity=entity, available_relations=relations, question=question,
                    history=history, max_selection_count=max_selection_count
                )
                # ... (rest of validation) ...
                llm_duration = time.monotonic() - start_llm_time
                logger.debug(f"Relation selection LLM call for '{entity}' completed in {llm_duration:.2f}s.")
                if selected is None: return []
                if not isinstance(selected, list): selected = [selected] if isinstance(selected, str) else []
                validated_selection = [s for s in selected if isinstance(s, str) and s.strip()]
                return validated_selection
            except Exception as e:
                 llm_duration = time.monotonic() - start_llm_time
                 logger.error(f"Exception during LLM relation selection for '{entity}' (after {llm_duration:.2f}s): {e}", exc_info=True)
                 return []

    def _execute_llm_attribute_selection(self,
                                          question: str, history: str,
                                          cvt_id: str, source_entity: str, source_relation: str,
                                          attributes: List[str], # List of "Rel -> Tgt" strings
                                          max_selection_count: int = 1) -> Optional[List[str]]:
         # (No change from previous version, but now called less often)
        if not attributes: return None
        with self.llm_semaphore:
            logger.debug(f"Acquired semaphore for CVT attribute selection LLM call (Rep CVT: {cvt_id}).") # Log uses representative CVT
            start_llm_time = time.monotonic()
            try:
                # NOTE: Using PREDICT model here, as per original correct logic
                selected = self.predict_model.select_attributes(
                    source_entity=source_entity, source_relation=source_relation, cvt_id=cvt_id,
                    available_attributes=attributes, question=question, history=history,
                    max_selection_count=max_selection_count
                )
                # ... (rest of validation) ...
                llm_duration = time.monotonic() - start_llm_time
                logger.debug(f"CVT attribute selection LLM call for group (Rep CVT: {cvt_id}) completed in {llm_duration:.2f}s.")
                if selected is None: return None
                if not isinstance(selected, list): selected = [selected] if isinstance(selected, str) else []
                validated_selection = [s for s in selected if isinstance(s, str) and s.strip() and s in attributes]
                if not validated_selection: return None
                return validated_selection[:max_selection_count]
            except Exception as e:
                 llm_duration = time.monotonic() - start_llm_time
                 logger.error(f"Exception during LLM attribute selection execution/parsing for group (Rep CVT: {cvt_id}) (after {llm_duration:.2f}s): {e}", exc_info=True)
                 return None

    # --- REMOVED ---
    # def _expand_specific_cvt(...) - Logic is now in _expand_cvt_group