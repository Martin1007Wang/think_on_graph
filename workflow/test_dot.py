import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_embeddings_and_similarity(model, texts: list) -> tuple:
    """
    计算文本列表的嵌入并返回相似性矩阵。
    
    Args:
        model: SentenceTransformer 模型
        texts: 要嵌入的文本列表
    
    Returns:
        embeddings: 嵌入向量数组
        similarities: 余弦相似性矩阵
    """
    # 计算嵌入
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    # 计算余弦相似性
    similarities = cosine_similarity(embeddings)
    
    return embeddings, similarities

def analyze_dot_impact():
    """分析点号对嵌入和相似性的影响。"""
    # 初始化模型
    model = SentenceTransformer('msmarco-distilbert-base-tas-b')
    
    # 测试用例：包含点号和不包含点号的文本对
    test_cases = [
        ("the.cat.is.on.the.mat.", "The cat is on the mat"),
        ("the.cat.is.on.the.mat.", "The dog is on the mat"),
        ("the cat is on the mat.", "The cat is on the mat"),
        ("the cat is on the mat.", "The dog is on the mat"),
    ]
    
    logger.info("开始分析点号对嵌入和相似性的影响...")
    
    for i, (text_with_dot, text_without_dot) in enumerate(test_cases, 1):
        logger.info(f"\n用例 {i}:")
        logger.info(f"  带点号文本: '{text_with_dot}'")
        logger.info(f"  无点号文本: '{text_without_dot}'")
        
        # 计算嵌入和相似性
        texts = [text_with_dot, text_without_dot]
        embeddings, similarities = compute_embeddings_and_similarity(model, texts)
        
        # 输出嵌入向量的 L2 范数差异
        emb_diff = np.linalg.norm(embeddings[0] - embeddings[1])
        logger.info(f"  嵌入向量 L2 范数差异: {emb_diff:.6f}")
        
        # 输出余弦相似性
        similarity = similarities[0, 1]
        logger.info(f"  余弦相似性: {similarity:.6f}")
        
        # 检查嵌入是否完全相同
        if np.array_equal(embeddings[0], embeddings[1]):
            logger.info("  嵌入向量完全相同")
        else:
            logger.info("  嵌入向量存在差异")

def main():
    """主函数"""
    try:
        analyze_dot_impact()
    except Exception as e:
        logger.error(f"发生错误: {e}")

if __name__ == "__main__":
    main()