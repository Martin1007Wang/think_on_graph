import os
import sys
import logging
import torch
import gc
import time
from memory_profiler import profile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your classes
from src.knowledge_graph import KnowledgeGraph
from src.llms import get_model
from src.knowledge_explorer import KnowledgeExplorer

# Memory tracking function
def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"[{tag}] GPU Memory: Allocated={allocated:.4f}GB, Reserved={reserved:.4f}GB")
        return allocated, reserved
    return 0, 0

# Create a test function that will use your code
@profile
def run_test():
    # Initialize your model
    model_name = "Llama-3.1-8B-Instruct"  # Adjust to your actual model name
    log_gpu_memory("Before model load")
    model = get_model(model_name)
    log_gpu_memory("After model load")
    
    # Initialize knowledge graph
    kg = KnowledgeGraph()
    
    # Initialize explorer
    explorer = KnowledgeExplorer(kg, model)
    log_gpu_memory("After explorer init")
    
    # Test data
    test_data = {
        "question": "Who directed Inception?",
        "id": "test_1",
        "q_entity": ["Christopher Nolan"],
        "answer": "Christopher Nolan"
    }
    processed_ids = set()
    
    # Process the question
    logger.info("Starting question processing")
    log_gpu_memory("Before processing")
    result = explorer.process_question(test_data, processed_ids)
    log_gpu_memory("After processing")
    logger.info("Completed question processing")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(1)  # Give time for memory to be released
    log_gpu_memory("After cleanup")
    
    return result

# Test with multiple questions to see if memory accumulates
@profile
def run_multiple_tests(num_tests=3):
    # Initialize your model
    model_name = "Llama-3.1-8B-Instruct"  # Adjust to your actual model name
    model = get_model(model_name)
    
    # Initialize knowledge graph
    kg = KnowledgeGraph()
    
    # Initialize explorer
    explorer = KnowledgeExplorer(kg, model)
    
    # Test data templates
    test_questions = [
        "Who directed Inception?",
        "What is the capital of France?",
        "Who wrote Harry Potter?"
    ]
    
    processed_ids = set()
    
    for i in range(num_tests):
        # Use modulo to cycle through questions
        question = test_questions[i % len(test_questions)]
        
        test_data = {
            "question": question,
            "id": f"test_{i}",
            "q_entity": ["Test Entity"],
            "answer": "Test Answer"
        }
        
        logger.info(f"\n--- Processing question {i+1}/{num_tests} ---")
        log_gpu_memory(f"Before question {i+1}")
        
        # Process the question
        result = explorer.process_question(test_data, processed_ids)
        
        log_gpu_memory(f"After question {i+1}")
        
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)  # Give time for memory to be released
        
        log_gpu_memory(f"After cleanup {i+1}")

if __name__ == "__main__":
    # Choose which test to run
    if len(sys.argv) > 1 and sys.argv[1] == "multiple":
        run_multiple_tests()
    else:
        # Run the single test
        result = run_test()
        
        # Print the result summary
        print("\nResult Summary:")
        if isinstance(result, dict):
            print(f"Question ID: {result.get('id', 'unknown')}")
            print(f"Answer found: {result.get('answer_found', False)}")
            if 'final_answer' in result and isinstance(result['final_answer'], dict):
                print(f"Can answer: {result['final_answer'].get('can_answer', False)}")
                print(f"Answer: {result['final_answer'].get('answer_sentence', 'No answer')}")
