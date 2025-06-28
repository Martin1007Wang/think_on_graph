import random
import logging
from collections import deque, defaultdict # defaultdict not currently used, but fine to keep
from typing import List, Tuple, Set, Dict, Optional, Any
from src.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

class PathGenerator:
    def __init__(
        self,
        kg: KnowledgeGraph,
        max_path_length: int = 3,
        top_k_relations: int = 5, # Used as default for semantic relation fetching
    ):
        self.kg = kg
        self.max_path_length = max_path_length
        self.default_top_k_relations = top_k_relations # Max relations to consider per step
        
    def get_shortest_paths(self, start_entity: str, answer_entity: str) -> List[List[Tuple[str, str, str]]]:
        if not start_entity or not answer_entity:
            logger.debug("get_shortest_paths: empty start_entity or answer_entity.")
            return []
        try:
            # max_depth in KG's get_shortest_paths is inclusive of number of relationships
            shortest_paths_from_kg = self.kg.get_shortest_paths(
                start_entity, 
                answer_entity, 
                max_depth=self.max_path_length 
            )
            return shortest_paths_from_kg if shortest_paths_from_kg else []
        except Exception as e:
            logger.error(f"Error in kg.get_shortest_paths for '{start_entity}'->'{answer_entity}': {e}", exc_info=True)
            return []

    # get_semantic_path and _expand_for_semantic_path are removed as they are not
    # currently used by the prepare_paths.py workflow based on previous modifications.
    # If you have other uses for them, they can be kept or re-added.

    def _perturb_relations_q_guided(self,
                                   positive_path: List[Tuple[str, str, str]],
                                   question: str,
                                   answer_entity: str,
                                   max_negatives_needed: int,
                                   relation_cache: Dict[str, List[Tuple[str, float]]] # Pass cache for reuse
                                  ) -> Set[Tuple[Tuple[str, str, str], ...]]:
        """
        Generates negative paths by perturbing relations in a positive path, guided by question semantics.
        Returns a set of unique negative paths (as tuples of tuples).
        """
        if not positive_path or max_negatives_needed <= 0:
            return set()

        generated_neg_path_structures: Set[Tuple[Tuple[str, str, str], ...]] = set()
        
        # Iterate from last hop backwards to prioritize changes near the end
        # Consider perturbing up to N hops, or all hops if time permits
        hops_to_consider_perturbing = list(range(len(positive_path) -1, -1, -1))
        random.shuffle(hops_to_consider_perturbing) # Add randomness to hop selection for perturbation

        for hop_idx in hops_to_consider_perturbing:
            if len(generated_neg_path_structures) >= max_negatives_needed:
                break

            src_entity, pos_relation, original_target_at_hop = positive_path[hop_idx]

            if src_entity not in relation_cache:
                try:
                    # Fetch more relations than default_top_k if we plan to sample more diverse ones
                    # Using default_top_k_relations as the top_k for the KG call.
                    relation_cache[src_entity] = self.kg.get_related_relations_by_question(
                        src_entity, question, top_k=self.default_top_k_relations * 2 # Fetch a bit more
                    )
                except Exception as e:
                    logger.debug(f"NegGen(RelPerturb): Error kg.get_related_relations_by_question for '{src_entity}': {e}")
                    relation_cache[src_entity] = []
            
            alt_relations_with_scores = [
                (rel, score) for rel, score in relation_cache.get(src_entity, []) if rel != pos_relation
            ]
            if not alt_relations_with_scores:
                continue
            
            # Try a few top alternative relations for this specific hop perturbation
            num_rels_to_try_this_hop = min(
                self.default_top_k_relations, # How many different relations to try at this hop
                len(alt_relations_with_scores),
                max_negatives_needed - len(generated_neg_path_structures) # Global cap
            )
            if num_rels_to_try_this_hop <=0: continue

            # Take the top scoring alternative relations
            for i in range(num_rels_to_try_this_hop):
                if len(generated_neg_path_structures) >= max_negatives_needed: break
                
                new_relation, _ = alt_relations_with_scores[i]
                
                try:
                    potential_new_tails = self.kg.get_target_entities(src_entity, new_relation, direction="out")
                except Exception as e:
                    logger.debug(f"NegGen(RelPerturb): Error kg.get_target_entities for '{src_entity}' via '{new_relation}': {e}")
                    continue
                
                if not potential_new_tails: continue

                valid_new_tails = [
                    t for t in potential_new_tails
                    if t != answer_entity and t != original_target_at_hop
                ]
                
                if not valid_new_tails: continue
                
                new_tail_entity = random.choice(valid_new_tails)
                
                path_prefix_tuples = tuple(positive_path[:hop_idx])
                new_hop_tuple = (src_entity, new_relation, new_tail_entity)
                current_negative_path_tuple = path_prefix_tuples + (new_hop_tuple,)
                
                generated_neg_path_structures.add(current_negative_path_tuple)
        
        return generated_neg_path_structures

    def _generate_truncated_paths(self,
                                 positive_path: List[Tuple[str, str, str]],
                                 answer_entity: str, # To ensure truncated path doesn't accidentally become the answer
                                 max_negatives_needed: int
                                ) -> Set[Tuple[Tuple[str, str, str], ...]]:
        """
        Generates negative paths by truncating a positive path at various lengths.
        A truncated path is meaningful if its endpoint is not the answer_entity.
        """
        if not positive_path or max_negatives_needed <= 0 or len(positive_path) < 1: # Min positive path length for truncation is 1
            return set()

        generated_neg_path_structures: Set[Tuple[Tuple[str, str, str], ...]] = set()
        
        # Truncate from length 1 up to length L-1
        # A path of length 0 (just start entity) is not usually a useful path representation here.
        # If positive_path has length L_pos (L_pos hops), it has L_pos+1 entities.
        # A truncated path of k hops must have k < L_pos.
        # Example: A-r1->B-r2->C (Length 2 path)
        # Truncated 1: A-r1->B
        
        # Iterate through possible truncation points (i.e., number of hops in the truncated path)
        for k_hops in range(1, len(positive_path)): # Truncated path has k_hops, from 1 to L-1 hops
            if len(generated_neg_path_structures) >= max_negatives_needed:
                break
            
            truncated_path_list = positive_path[:k_hops] # Takes first k_hops elements
            
            # Ensure the truncated path is not empty and its tail is not the answer entity
            # (though if positive_path is a shortest path to answer_entity, intermediate tails shouldn't be answer_entity)
            if truncated_path_list:
                # The last entity of the truncated path
                # last_hop_in_truncated = truncated_path_list[-1] -> (src, rel, tail)
                # tail_of_truncated = last_hop_in_truncated[2]
                # if tail_of_truncated != answer_entity: # This check is implicitly true for shortest paths usually
                generated_neg_path_structures.add(tuple(truncated_path_list))
        
        return generated_neg_path_structures

    def get_negative_paths(self, 
                           positive_path: List[Tuple[str, str, str]], 
                           question: str, 
                           answer_entity: str, 
                           max_negatives_per_positive: int = 5,
                           # Allow specifying strategy weights or order in future if needed
                           enable_relation_perturbation: bool = True,
                           enable_truncation: bool = True 
                          ) -> List[List[Tuple[str, str, str]]]:
        """
        Generates a list of negative paths for a given positive path using multiple strategies.
        """
        if not positive_path:
            logger.debug("get_negative_paths called with empty positive_path.")
            return []
        
        # Using a set to store unique negative paths (as tuples of tuples)
        all_unique_negative_path_tuples: Set[Tuple[Tuple[str, str, str], ...]] = set()
        
        # Shared cache for KG calls within this function call for this positive_path
        # Primarily for get_related_relations_by_question
        relation_scores_cache_for_this_call: Dict[str, List[Tuple[str, float]]] = {}

        # Strategy 1: Relation Perturbation (Question Guided)
        if enable_relation_perturbation and len(all_unique_negative_path_tuples) < max_negatives_per_positive:
            negs_from_rel_perturb = self._perturb_relations_q_guided(
                positive_path, question, answer_entity,
                max_negatives_needed = max_negatives_per_positive - len(all_unique_negative_path_tuples),
                relation_cache = relation_scores_cache_for_this_call
            )
            all_unique_negative_path_tuples.update(negs_from_rel_perturb)

        # Strategy 2: Path Truncation
        if enable_truncation and len(all_unique_negative_path_tuples) < max_negatives_per_positive:
            if len(positive_path) > 1: # Truncation only makes sense if path has at least 2 hops to make a shorter one of at least 1 hop
                negs_from_truncation = self._generate_truncated_paths(
                    positive_path, answer_entity,
                    max_negatives_needed = max_negatives_per_positive - len(all_unique_negative_path_tuples)
                )
                all_unique_negative_path_tuples.update(negs_from_truncation)
            
        # Convert set of tuple-paths back to list of list-paths
        final_negative_paths_list_of_lists = [list(p_struct) for p_struct in all_unique_negative_path_tuples]
        
        # Ensure we don't exceed max_negatives_per_positive
        # Shuffle to get a random sample if more were generated than needed.
        if len(final_negative_paths_list_of_lists) > max_negatives_per_positive:
            random.shuffle(final_negative_paths_list_of_lists)
            return final_negative_paths_list_of_lists[:max_negatives_per_positive]
            
        return final_negative_paths_list_of_lists