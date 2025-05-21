# Necessary imports (ensure these align with your project structure)
import logging
import threading
import time
from typing import Optional, List, Dict, Any, Tuple, Set, ClassVar, FrozenSet
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
# Assuming these imports are correctly set up in your project
try:
    from src.knowledge_graph import KnowledgeGraph
    from src.model_interface import ModelInterface # Assumes uses LLMOutputParserOptimized
    from src.path_manager import PathManager
    from src.utils.data_utils import EntityExpansion, EntityRelation
except ImportError:
    # Dummy classes for standalone execution/testing
    class KnowledgeGraph:
        def get_related_relations(self, node, direction): return []
        def get_target_entities(self, entity, relation, direction): return []
    class ModelInterface:
        def select_relations(self, **kwargs): return []
        def select_attributes(self, **kwargs): return None # Important for CVT path
        # Dummy parser attribute for init check
        class DummyParser:
            def parse_attribute_selection(self, output, **kwargs): return None
        parser = DummyParser()

    class PathManager:
        def is_coded_entity(self, entity: str) -> bool: return entity.startswith("CVT_") or entity.startswith("m.") # Example dummy logic
    class EntityExpansion:
        def __init__(self, entity): self.entity = entity; self.relations: List[EntityRelation] = []
    class EntityRelation:
        def __init__(self, relation, targets, metadata=None): self.relation = relation; self.targets = targets; self.metadata = metadata if metadata else {}

logger = logging.getLogger(__name__)
# Basic config if not set externally
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')


class EntityExpander:
    RELATION_SELECTION_TEMPLATE: ClassVar[str] = "relation_selection"
    RELATION_SELECTION_WITH_HISTORY_TEMPLATE: ClassVar[str] = "relation_selection_history"
    CVT_ATTRIBUTE_SELECTION_TEMPLATE: ClassVar[str] = "cvt_attribute_selection" # Template name used by explore_model
    CVT_ATTRIBUTE_PARSER_METHOD_NAME: ClassVar[str] = "parse_attribute_selection" # Method explore_model's parser MUST have

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
        if min_cvts_for_parallel_processing < 1: raise ValueError("min_cvts_for_parallel_processing must be >= 1")

        self.kg = kg
        self.explore_model = explore_model_interface
        self.predict_model = predict_model_interface # Correct model for attribute selection
        self.path_manager = path_manager
        self.max_relation_selection_count = max_relation_selection_count # Max relations to explore from an entity
        self.max_workers = max_workers
        self.llm_semaphore = threading.Semaphore(llm_concurrency_limit)
        self.min_cvts_for_parallel_processing = min_cvts_for_parallel_processing

        # --- Initialization Checks ---
        parser_instance = getattr(self.predict_model, 'parser', None)
        if not parser_instance or not hasattr(parser_instance, self.CVT_ATTRIBUTE_PARSER_METHOD_NAME):
            logger.error(f"CRITICAL: LLMOutputParser used by predict_model_interface does NOT have the required '{self.CVT_ATTRIBUTE_PARSER_METHOD_NAME}' method. CVT attribute selection parsing WILL FAIL.")
        else:
            logger.info(f"CVT attribute selection parser '{self.CVT_ATTRIBUTE_PARSER_METHOD_NAME}' found on predict_model.")

        if not hasattr(self.predict_model, 'select_attributes'):
             logger.warning(f"Predict model interface ({type(self.predict_model).__name__}) might not have 'select_attributes' method.")
        elif not callable(getattr(self.predict_model, 'select_attributes', None)):
             logger.warning(f"Predict model interface has 'select_attributes', but it is not callable.")

        logger.info(f"EntityExpander initialized with max_workers={self.max_workers}, llm_concurrency_limit={llm_concurrency_limit}, max_relation_selection_count={self.max_relation_selection_count}, min_cvts_for_parallel_processing applied to GROUPS={self.min_cvts_for_parallel_processing}")

    def _fetch_relations(self, node: str) -> List[str]:
        try:
            relations = self.kg.get_related_relations(node, "out")
            if not relations: return []
            cleaned_relations = [str(r).strip() for r in relations if r is not None]
            return [r for r in cleaned_relations if r]
        except Exception as e:
            logger.error(f"Error getting relations for node '{node}': {e}", exc_info=True)
            return []

    def expand_entity(self, entity: str, question: str, history: str) -> Optional[EntityExpansion]:
        logger.debug(f"Expanding entity: '{entity}' for question: '{question[:50]}...'")
        start_time = time.monotonic()
        all_relations = self._fetch_relations(entity)
        if not all_relations: return None

        selected_relations: List[str] = []
        try:
            selected_relations = self._execute_llm_relation_selection( entity=entity, relations=all_relations, question=question, history=history, max_selection_count=self.max_relation_selection_count )
        except Exception as e: logger.error(f"Error during relation selection for '{entity}': {e}", exc_info=True)
        if not selected_relations: return None
        logger.debug(f"LLM selected {len(selected_relations)} relations for '{entity}': {selected_relations}")

        expansion_result = EntityExpansion(entity=entity)
        potential_cvt_tasks: List[Tuple[str, str, str, str]] = []
        for relation in selected_relations:
            try:
                targets = self.kg.get_target_entities(entity, relation, "out")
                if not targets: continue
                target_list = list(targets) if isinstance(targets, (list, tuple, set)) else [targets]
                named_entities, coded_entities = self._separate_entities(target_list)
                if named_entities: expansion_result.relations.append(EntityRelation(relation=relation, targets=named_entities))
                if coded_entities:
                    history_to_cvt_base = f"{history} --> {entity} --[{relation}]-->"
                    for cvt_id in coded_entities: potential_cvt_tasks.append((relation, cvt_id, question, f"{history_to_cvt_base}{cvt_id}"))
            except Exception as e: logger.error(f"Error processing relation '{relation}' targets for '{entity}': {e}", exc_info=True)

        grouped_tasks = defaultdict(list)
        structure_cache: Dict[str, Optional[Dict[str, List[str]]]] = {}
        logger.debug(f"Grouping {len(potential_cvt_tasks)} potential CVT tasks...")
        for source_relation, cvt_id, q, hist_to_cvt in potential_cvt_tasks:
            try:
                if cvt_id in structure_cache:
                    cvt_structure = structure_cache[cvt_id]
                    if cvt_structure is None: continue
                else:
                    cvt_structure = self._fetch_cvt_outgoing_structure(cvt_id)
                    structure_cache[cvt_id] = cvt_structure
                if not cvt_structure: continue

                canonical_structure_items = []
                for out_rel, out_targets in sorted(cvt_structure.items()):
                    cleaned_targets = [str(t).strip() for t in out_targets if t and str(t).strip()]
                    if cleaned_targets: canonical_structure_items.append((out_rel, frozenset(sorted(cleaned_targets))))
                if not canonical_structure_items: continue
                group_key = (source_relation, frozenset(canonical_structure_items))
                grouped_tasks[group_key].append((cvt_id, q, hist_to_cvt))
            except Exception as group_e: logger.error(f"Error fetching structure or grouping CVT '{cvt_id}': {group_e}", exc_info=True)

        num_cvt_groups = len(grouped_tasks)
        if num_cvt_groups > 0:
            logger.info(f"Grouped {len(potential_cvt_tasks)} potential tasks into {num_cvt_groups} distinct CVT structure groups.")
            processed_group_count = 0
            use_parallel_group_processing = num_cvt_groups >= self.min_cvts_for_parallel_processing
            tasks_to_process = list(grouped_tasks.items())
            group_results: List[EntityRelation] = []

            if use_parallel_group_processing:
                actual_workers = min(num_cvt_groups, self.max_workers)
                logger.debug(f"Processing {num_cvt_groups} CVT groups in parallel using {actual_workers} workers.")
                with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers, thread_name_prefix=f'ExpandCVTGroup_{entity[:5]}_') as executor:
                    future_to_group_key = { executor.submit(self._expand_cvt_group, entity, group_key, group_tasks): group_key for group_key, group_tasks in tasks_to_process }
                    for future in concurrent.futures.as_completed(future_to_group_key):
                        group_key = future_to_group_key[future]; processed_group_count += 1
                        try:
                            group_expansion_relation = future.result()
                            if group_expansion_relation: group_results.append(group_expansion_relation)
                        except Exception as e: logger.error(f"Error processing parallel result for CVT group (KeyRel: {group_key[0]}): {e}", exc_info=True)
            else:
                logger.debug(f"Processing {num_cvt_groups} CVT groups sequentially.")
                for group_key, group_tasks in tasks_to_process:
                    processed_group_count += 1
                    try:
                        group_expansion_relation = self._expand_cvt_group(entity, group_key, group_tasks)
                        if group_expansion_relation: group_results.append(group_expansion_relation)
                    except Exception as e: logger.error(f"Error during sequential expansion of CVT group (KeyRel: {group_key[0]}): {e}", exc_info=True)

            if group_results: expansion_result.relations.extend(group_results)
            logger.info(f"Finished processing {processed_group_count}/{num_cvt_groups} CVT groups for '{entity}'.")

        final_relation_count = 0
        if expansion_result.relations:
            original_count = len(expansion_result.relations)
            expansion_result = self._deduplicate_results(expansion_result)
            final_relation_count = len(expansion_result.relations)
            if original_count > final_relation_count: logger.debug(f"Deduplication reduced paths from {original_count} to {final_relation_count} for '{entity}'.")
        duration = time.monotonic() - start_time
        logger.debug(f"Finished expanding entity '{entity}'. Found {final_relation_count} unique paths. Duration: {duration:.2f}s")
        return expansion_result if final_relation_count > 0 else None

    def _fetch_cvt_outgoing_structure(self, cvt_id: str) -> Optional[Dict[str, List[str]]]:
        logger.debug(f"Fetching structure for CVT '{cvt_id}'")
        details: Dict[str, List[str]] = {}
        try:
            relations = self.kg.get_related_relations(cvt_id, "out")
            if not relations: return {}
            for rel in relations:
                rel_key = str(rel).strip();
                if not rel_key: continue
                targets = self.kg.get_target_entities(cvt_id, rel_key, "out")
                if targets:
                    target_list = list(targets) if isinstance(targets, (list, tuple, set)) else [targets]
                    valid_targets = [str(t).strip() for t in target_list if t is not None and str(t).strip()]
                    if valid_targets: details[rel_key] = valid_targets
            return details
        except Exception as kg_e:
            logger.warning(f"Failed to fetch structure for CVT '{cvt_id}': {kg_e}", exc_info=False)
            return None

    def _expand_cvt_group(self, source_entity: str, group_key: Tuple[str, FrozenSet[Tuple[str, FrozenSet[str]]]], group_tasks: List[Tuple[str, str, str]]) -> Optional[EntityRelation]:
        if not group_tasks: return None
        source_relation, canonical_structure = group_key
        representative_task = group_tasks[0]; representative_cvt_id, question, history = representative_task
        list_of_cvt_ids = [task[0] for task in group_tasks]
        logger.debug(f"Expanding CVT group via '{source_relation}' (Rep CVT: {representative_cvt_id}, {len(list_of_cvt_ids)} CVTs)")

        available_attributes: List[str] = []
        seen_attributes: Set[str] = set()
        for rel_str, targets_fs in canonical_structure:
            for target_str in targets_fs:
                attribute = f"{rel_str} -> {target_str}"
                if attribute not in seen_attributes: available_attributes.append(attribute); seen_attributes.add(attribute)
        if not available_attributes: logger.error(f"Internal Error: No attributes for LLM selection from group (Rep CVT: {representative_cvt_id})"); return None

        selected_attribute_strings: Optional[List[str]] = None
        try:
            selected_attribute_strings = self._execute_llm_attribute_selection(question=question, history=history, cvt_id=representative_cvt_id, source_entity=source_entity, source_relation=source_relation, attributes=available_attributes, max_selection_count=1)
        except Exception as llm_e: logger.error(f"Error calling LLM attribute selection helper for group (Rep CVT: {representative_cvt_id}): {llm_e}", exc_info=True); return None
        if not selected_attribute_strings: logger.debug(f"LLM selected no attribute from CVT group (Rep CVT: {representative_cvt_id}). Path terminates."); return None
        selected_string = selected_attribute_strings[0]

        try:
            parts = selected_string.split(" -> ", 1)
            if len(parts) == 2:
                selected_relation, selected_target = parts[0].strip(), parts[1].strip()
                if not selected_relation or not selected_target: logger.warning(f"Parsed empty rel/tgt from '{selected_string}' for group (Rep CVT: {representative_cvt_id})."); return None
                logger.debug(f"LLM selected '{selected_relation}' -> '{selected_target}' from group (Rep CVT: {representative_cvt_id}).")

                # OPTIMIZED FORMAT: Use representative CVT ID, no [Group:N] tag
                two_hop_path_key = f"{source_relation} > {representative_cvt_id} > {selected_relation}"
                normalized_path_key = f"{source_relation}>{selected_relation}"

                return EntityRelation(
                    relation=two_hop_path_key, targets=[selected_target],
                    metadata={ "is_cvt_path": True, "is_cvt_group": True, "merged_cvt_ids": sorted(list_of_cvt_ids), "normalized_path": normalized_path_key, "source_relation": source_relation, "selected_relation": selected_relation, "selected_target": selected_target, "representative_cvt_id": representative_cvt_id }
                )
            else: logger.error(f"Could not parse selected attribute string '{selected_string}' for group."); return None
        except Exception as parse_e: logger.error(f"Error parsing LLM selection result ('{selected_string}') for group: {parse_e}", exc_info=True); return None

    def _deduplicate_results(self, expansion: EntityExpansion) -> EntityExpansion:
        if not expansion or not expansion.relations: return expansion
        unique_relations_map: Dict[Tuple[str, str], EntityRelation] = {}
        for relation_obj in expansion.relations:
            is_cvt_path = relation_obj.metadata and relation_obj.metadata.get("is_cvt_path", False)
            path_key_part = relation_obj.metadata["normalized_path"] if is_cvt_path and "normalized_path" in relation_obj.metadata else relation_obj.relation
            for target in relation_obj.targets:
                target_str = str(target); unique_key = (path_key_part, target_str)
                if unique_key not in unique_relations_map:
                    if len(relation_obj.targets) > 1: unique_relations_map[unique_key] = EntityRelation(relation=relation_obj.relation, targets=[target], metadata=relation_obj.metadata.copy() if relation_obj.metadata else {})
                    else: unique_relations_map[unique_key] = relation_obj
                else:
                    existing_relation = unique_relations_map[unique_key]
                    if is_cvt_path and existing_relation.metadata and existing_relation.metadata.get("is_cvt_path"):
                        current_ids = set(relation_obj.metadata.get("merged_cvt_ids", []))
                        existing_ids = set(existing_relation.metadata.get("merged_cvt_ids", []))
                        merged_set = existing_ids.union(current_ids)
                        if len(merged_set) > len(existing_ids): existing_relation.metadata["merged_cvt_ids"] = sorted(list(merged_set))
        deduplicated_expansion = EntityExpansion(entity=expansion.entity)
        deduplicated_expansion.relations = list(unique_relations_map.values())
        return deduplicated_expansion

    def _separate_entities(self, entities: List[Any]) -> Tuple[List[str], List[str]]:
        if not entities: return [], []
        is_coded = self.path_manager.is_coded_entity
        named_set, coded_set = set(), set()
        for entity_obj in entities:
            try:
                entity_str = str(entity_obj).strip()
                if entity_str: (coded_set if is_coded(entity_str) else named_set).add(entity_str)
            except Exception as e: logger.warning(f"Error processing target entity '{entity_obj!r}': {e}", exc_info=False)
        return sorted(list(named_set)), sorted(list(coded_set))

    def _execute_llm_relation_selection(self, entity: str, relations: List[str], question: str, history: str, max_selection_count: int) -> List[str]:
        if not relations: return []
        with self.llm_semaphore:
            logger.debug(f"Semaphore acquired for relation selection (Entity: {entity})")
            try:
                if not hasattr(self.explore_model, 'select_relations') or not callable(getattr(self.explore_model, 'select_relations')):
                     logger.error(f"Explore model missing 'select_relations'."); return []
                selected = self.explore_model.select_relations(entity=entity, available_relations=relations, question=question, history=history, max_selection_count=max_selection_count)
                if selected is None: return []
                if not isinstance(selected, list): selected = [selected] if isinstance(selected, str) else []
                validated = [s.strip() for s in selected if isinstance(s, str) and s.strip() and s.strip() in relations]
                return validated[:max_selection_count]
            except Exception as e: logger.error(f"Exception during LLM relation selection for '{entity}': {e}", exc_info=True); return []
            finally: logger.debug(f"Semaphore released for relation selection (Entity: {entity})")

    def _execute_llm_attribute_selection(self, question: str, history: str, cvt_id: str, source_entity: str, source_relation: str, attributes: List[str], max_selection_count: int = 1) -> Optional[List[str]]:
        if not attributes: return None
        with self.llm_semaphore:
            logger.debug(f"Semaphore acquired for CVT attribute selection (Rep CVT: {cvt_id})")
            try:
                if not hasattr(self.explore_model, 'select_attributes') or not callable(getattr(self.explore_model, 'select_attributes')):
                    logger.error(f"Predict model missing 'select_attributes'."); return None
                selected = self.explore_model.select_attributes(source_entity=source_entity, source_relation=source_relation, cvt_id=cvt_id, available_attributes=attributes, question=question, history=history, max_selection_count=max_selection_count)
                if selected is None: return None
                if not isinstance(selected, list): selected = [selected] if isinstance(selected, str) else []
                validated = [s.strip() for s in selected if isinstance(s, str) and s.strip() and s.strip() in attributes]
                return validated[:max_selection_count] if validated else None
            except Exception as e: logger.error(f"Exception during LLM attribute selection for group (Rep CVT: {cvt_id}): {e}", exc_info=True); return None
            finally: logger.debug(f"Semaphore released for CVT attribute selection (Rep CVT: {cvt_id})")