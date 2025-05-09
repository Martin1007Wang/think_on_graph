import random
import logging
from collections import deque, defaultdict
from typing import List, Tuple, Set, Dict, Optional, Any

from src.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

class PathGenerator:
    def __init__(
        self,
        kg: KnowledgeGraph, # Your KnowledgeGraph class instance
        max_path_length: int = 3,
        top_k_relations: int = 5,
    ):
        self.kg = kg
        self.max_path_length = max_path_length
        self.default_top_k_relations = top_k_relations
        
    def get_shortest_paths(self, start_entity: str, answer_entity: str) -> List[List[Tuple[str, str, str]]]:
        """
        Gets all shortest paths between start_entity and answer_entity from the KnowledgeGraph.
        The KG method is expected to return paths as List[List[Tuple[str, str, str]]].
        """
        if not start_entity or not answer_entity:
            logger.warning("get_shortest_paths called with empty start_entity or answer_entity.")
            return []
        try:
            shortest_paths_from_kg = self.kg.get_shortest_paths(
                start_entity, 
                answer_entity, 
                max_depth=self.max_path_length 
            )
            return shortest_paths_from_kg if shortest_paths_from_kg else []
        except Exception as e:
            logger.error(f"Error calling kg.get_shortest_paths for '{start_entity}' to '{answer_entity}': {e}", exc_info=True)
            return []

    def get_semantic_path(self, start_entity: str, answer_entity: str, question: str, 
                          max_top_k_exploration: int = 20
                         ) -> Tuple[Optional[List[Tuple[str, str, str]]], Set[str]]:
        """
        Finds a semantically guided path from start_entity to answer_entity.
        Iteratively widens the search (top_k relations considered at each step).
        Returns a tuple: (path_if_found_else_None, set_of_all_visited_entities).
        """
        if not start_entity or not answer_entity or not question:
            logger.warning("get_semantic_path called with empty start_entity, answer_entity, or question.")
            return None, set()

        all_visited_entities_globally: Set[str] = {start_entity}
        
        # Cache for relation scores within this single call to get_semantic_path
        # Key: entity_id (str), Value: List[Tuple[str, float]] (relation, score)
        call_specific_relation_scores_cache: Dict[str, List[Tuple[str, float]]] = {}

        for current_top_k in range(self.default_top_k_relations, max_top_k_exploration + 1, self.default_top_k_relations):
            logger.debug(f"Semantic path search: trying with top_k = {current_top_k}")
            
            # For each top_k, perform a fresh BFS-like search
            visited_paths_for_this_top_k: Dict[str, List[Tuple[str, str, str]]] = {start_entity: []} 
            queue: deque = deque([(start_entity, [])]) 
            
            path_to_target = self._expand_for_semantic_path(
                queue, visited_paths_for_this_top_k, answer_entity, question, 
                top_k=current_top_k, max_depth=self.max_path_length,
                relation_scores_cache=call_specific_relation_scores_cache # Pass the cache
            )
            
            all_visited_entities_globally.update(visited_paths_for_this_top_k.keys())

            if path_to_target is not None:
                logger.info(f"Semantic path found to '{answer_entity}' with top_k={current_top_k}, length {len(path_to_target)}.")
                return path_to_target, all_visited_entities_globally
            
            logger.debug(f"Semantic path to '{answer_entity}' not found with top_k = {current_top_k}. Queue exhausted for this top_k.")

        logger.info(f"Semantic path to '{answer_entity}' not found after exploring up to top_k={max_top_k_exploration}.")
        return None, all_visited_entities_globally

    def _expand_for_semantic_path(self, 
                                  queue: deque, 
                                  visited_paths: Dict[str, List[Tuple[str, str, str]]], 
                                  target_entity: str, 
                                  question: str, 
                                  top_k: int, 
                                  max_depth: int,
                                  relation_scores_cache: Dict[str, List[Tuple[str, float]]] # Added cache parameter
                                 ) -> Optional[List[Tuple[str, str, str]]]:
        """
        Expands nodes from the queue for semantic path search for a GIVEN top_k.
        Uses a cache for relation scores to avoid redundant KG calls.
        """
        while queue: 
            current_entity, current_path = queue.popleft()
            
            if len(current_path) >= max_depth:
                continue
            
            try:
                # Use cache for related relation scores
                if current_entity in relation_scores_cache:
                    related_relation_scores = relation_scores_cache[current_entity]
                    logger.debug(f"Cache hit for related relations for '{current_entity}'")
                else:
                    logger.debug(f"Cache miss for related relations for '{current_entity}'. Querying KG.")
                    related_relation_scores = self.kg.get_related_relations_by_question(current_entity, question)
                    relation_scores_cache[current_entity] = related_relation_scores # Store in cache
            except Exception as e:
                logger.error(f"Error in kg.get_related_relations_by_question for '{current_entity}': {e}", exc_info=True)
                continue 

            if not related_relation_scores:
                continue
                
            actual_top_k = min(top_k, len(related_relation_scores))
            candidate_relations_with_scores = related_relation_scores[:actual_top_k]
            
            for relation, score in candidate_relations_with_scores:
                try:
                    tail_entities = self.kg.get_target_entities(current_entity, relation, direction="out")
                except Exception as e:
                    logger.error(f"Error in kg.get_target_entities for '{current_entity}' via '{relation}': {e}", exc_info=True)
                    continue

                if not tail_entities:
                    continue
                    
                for tail_entity in tail_entities:
                    new_path = current_path + [(current_entity, relation, tail_entity)]

                    if tail_entity in visited_paths and len(visited_paths[tail_entity]) <= len(new_path):
                        continue 
                    
                    visited_paths[tail_entity] = new_path
                    
                    if tail_entity == target_entity:
                        return new_path 
                        
                    if len(new_path) < max_depth: 
                        queue.append((tail_entity, new_path))
        return None

    def get_negative_paths(self, 
                           positive_path: List[Tuple[str, str, str]], 
                           question: str, 
                           answer_entity: str, 
                           max_negatives_per_positive: int = 5 
                          ) -> List[List[Tuple[str, str, str]]]:
        if not positive_path:
            logger.warning("get_negative_paths called with empty positive_path.")
            return []
        
        negative_paths_found: List[List[Tuple[str, str, str]]] = []
        # This cache is specific to a single call of get_negative_paths, which is good.
        relation_cache: Dict[str, List[Tuple[str, float]]] = {} 

        for hop_idx in range(len(positive_path) - 1, -1, -1): # Iterate from last hop backwards
            if len(negative_paths_found) >= max_negatives_per_positive:
                break

            src_entity, positive_relation_at_hop, original_target_at_hop = positive_path[hop_idx]
            
            if src_entity not in relation_cache:
                try:
                    relation_cache[src_entity] = self.kg.get_related_relations_by_question(src_entity, question)
                except Exception as e:
                    logger.error(f"Error getting related relations for '{src_entity}' (neg path gen): {e}", exc_info=True)
                    relation_cache[src_entity] = [] # Ensure it's an empty list on error

            alternative_relations_with_scores = [
                (rel, score) for rel, score in relation_cache.get(src_entity, []) if rel != positive_relation_at_hop
            ]
            
            if not alternative_relations_with_scores:
                continue

            # Determine how many alternatives to actually try for this hop
            num_alternatives_to_try = min(
                max_negatives_per_positive - len(negative_paths_found), 
                len(alternative_relations_with_scores),
                self.default_top_k_relations # Cap alternatives considered from this hop to avoid too many KG calls from one point
            )
            
            # Consider top-N alternatives based on score (already sorted by KG)
            relations_to_try_at_this_hop = alternative_relations_with_scores[:num_alternatives_to_try]
            
            for new_relation, _ in relations_to_try_at_this_hop:
                if len(negative_paths_found) >= max_negatives_per_positive:
                    break 
                
                try:
                    potential_new_tails = self.kg.get_target_entities(src_entity, new_relation)
                except Exception as e:
                    logger.error(f"Error getting target entities for '{src_entity}' via '{new_relation}': {e}", exc_info=True)
                    continue

                if not potential_new_tails:
                    continue
                
                # Filter out tails that are the actual answer or the original target of this hop
                valid_new_tails = [
                    t for t in potential_new_tails 
                    if t != answer_entity and t != original_target_at_hop
                ]
                
                if not valid_new_tails:
                    continue
                
                # Choose one new tail (randomly or deterministically)
                new_tail_entity = random.choice(valid_new_tails) 
                # For determinism:
                # random.shuffle(valid_new_tails) # If KG returns ordered list, shuffle first
                # new_tail_entity = valid_new_tails[0]
                
                path_prefix = positive_path[:hop_idx]
                current_negative_path = list(path_prefix) + [(src_entity, new_relation, new_tail_entity)]
                
                negative_paths_found.append(current_negative_path)
                                        
        return negative_paths_found