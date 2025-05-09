import json
from typing import List, Dict, Any


def path_length(path: str) -> int:
    """计算路径的hop数（以分号分隔的步数）"""
    if not path:
        return 0
    return len(path.split(' ; '))


def find_long_semantic_examples(
    data: List[Dict[str, Any]],
    min_length_diff: int = 2
) -> List[Dict[str, Any]]:
    """
    筛选semantic_paths和shortest_paths不同，且semantic_paths显著长于shortest_paths的样本。
    显著长于定义为semantic_paths中最短路径长度 > shortest_paths中最短路径长度 + min_length_diff - 1。

    Args:
        data: 数据集列表。
        min_length_diff: 最小长度差，默认为2。

    Returns:
        满足条件的样本列表。
    """
    results = []
    for item in data:
        semantic_paths = item.get("semantic_paths", [])
        shortest_paths = item.get("shortest_paths", [])
        if not semantic_paths or not shortest_paths:
            continue
        # 判断内容是否不同
        if set(semantic_paths) == set(shortest_paths):
            continue
        # 计算最短路径长度
        min_semantic = min(path_length(p) for p in semantic_paths)
        min_shortest = min(path_length(p) for p in shortest_paths)
        if min_semantic > min_shortest + min_length_diff - 1:
            results.append({
                "id": item.get("id"),
                "question": item.get("question"),
                "semantic_paths": semantic_paths,
                "shortest_paths": shortest_paths,
                "min_semantic_length": min_semantic,
                "min_shortest_length": min_shortest
            })
    return results


def main() -> None:
    with open("/mnt/wangjingxiong/think_on_graph/data/processed/rmanluo_RoG-webqsp_train/path_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    results = find_long_semantic_examples(data)
    print(f"找到 {len(results)} 个semantic_paths显著长于shortest_paths的样本：\n")
    for item in results:
        print(f"ID: {item['id']}")
        print(f"Question: {item['question']}")
        print(f"semantic_paths (min len={item['min_semantic_length']}): {item['semantic_paths']}")
        print(f"shortest_paths (min len={item['min_shortest_length']}): {item['shortest_paths']}")
        print('-' * 80)


if __name__ == "__main__":
    main()
