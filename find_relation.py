import json

def verify_relation_in_chosen_field(file_path: str, target_relation: str):
    """
    检查 JSONL 文件中每一行的 'chosen' 字段是否包含目标关系字符串。

    Args:
        file_path (str): preference_data.jsonl 文件的路径。
        target_relation (str): 要搜索的目标关系字符串。
    """
    found_count = 0
    total_lines = 0
    
    print(f"🔍 开始扫描文件: '{file_path}'")
    print(f"🎯 目标关系: '{target_relation}'")
    print("-" * 50)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines = line_num
                try:
                    # 解析当前行
                    data = json.loads(line)
                    
                    # 获取 'chosen' 字段的值，使用 .get() 避免 KeyError
                    chosen_value = data.get('chosen')
                    
                    # 确保 'chosen' 字段存在且其值为字符串类型
                    if isinstance(chosen_value, str):
                        # 检查目标关系是否为 'chosen' 值的子字符串
                        if target_relation in chosen_value:
                            found_count += 1
                            print(f"✅ 在第 {line_num} 行找到目标关系。")
                            
                except json.JSONDecodeError:
                    print(f"⚠️ 警告: 第 {line_num} 行不是有效的 JSON 格式，已跳过。")
                except Exception as e:
                    print(f"❌ 在处理第 {line_num} 行时发生意外错误: {e}")

    except FileNotFoundError:
        print(f"❌ 错误: 文件 '{file_path}' 未找到。请检查文件路径是否正确。")
        return

    print("-" * 50)
    print("📊 扫描结果汇总:")
    print(f"总共处理行数: {total_lines}")
    print(f"在 'chosen' 字段中找到目标关系的总次数: {found_count}")
    print("✨ 脚本执行完毕。")


if __name__ == '__main__':
    # --- 参数配置 ---
    
    # 1. 指定您的数据文件路径
    PREFERENCE_FILE_PATH = '/mnt/wangjingxiong/think_on_graph/data/preference_dataset/train_cand_pn_only_pos_shortest_paths/preference_data.jsonl'
    
    # 2. 指定您要检验的关系字符串
    RELATION_TO_VERIFY = "location.statistical_region.gni_in_ppp_dollars-measurement_unit.dated_money_value.currency"
    
    # --- 执行验证 ---
    verify_relation_in_chosen_field(PREFERENCE_FILE_PATH, RELATION_TO_VERIFY)