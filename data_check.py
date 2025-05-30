import pandas as pd
import os
import shutil
import traceback

# --- 配置区域 ---
# 原始数据输入目录 (数据文件无表头，表头在对应的 .header.csv 文件中)
# --- 配置区域 ---
# 原始数据输入目录 (数据文件无表头，表头在对应的 .header.csv 文件中)
BASE_ORIGINAL_DATA_PATH = "/mnt/wangjingxiong/neo4j-freebase/data"
# 最终处理完毕，可以直接用于导入的文件的输出目录
FINAL_OUTPUT_PATH = "/mnt/wangjingxiong/neo4j-freebase/data/final_cleaned_v6" # 使用新的输出目录名
README_FILE = os.path.join(FINAL_OUTPUT_PATH, "readme_pipeline_v6.md")

# 文件配置
# For "foreign_keys" -> "source_id_col_in_source_file": This MUST be the actual ID column name
# in the header of the source node file (e.g., "lid:ID" for entities and types, "labelId:ID" for edge-labels)
CONFIG = {
    "entities": {
        "data_filename": "entities.csv",
        "header_filename": "entities.header.csv",
        "id_col": "lid:ID", # This is the primary ID column for entities
        "score_col": "score:FLOAT"
    },
    "types": {
        "data_filename": "types.csv",
        "header_filename": "types.header.csv",
        "id_col": "lid:ID" # This is the primary ID column for types
    },
    "edge-labels": {
        "data_filename": "edge-labels.csv",
        "header_filename": "edge-labels.header.csv",
        "id_col": "labelId:ID", # This is the primary ID column for edge-labels
        "rel_type_code_col": "code"
    },
    "other.edges": {
        "data_filename": "other.edges.csv",
        "header_filename": "other.edges.header.csv",
        "foreign_keys": [
            { "fk_col": ":START_ID", "source_node_key": "entities", "source_id_col_in_source_file": "lid:ID"},
            { "fk_col": ":END_ID",   "source_node_key": "entities", "source_id_col_in_source_file": "lid:ID"},
            { "fk_col": "labelId:LONG", "source_node_key": "edge-labels", "source_id_col_in_source_file": "labelId:ID"}
            # Note: new_fk_col_name for labelId:LONG (to become labelId) is handled implicitly by header if desired,
            # or you'd read the header, change it, and save with new header.
            # For simplicity, this version assumes labelId:LONG values will be mapped to edge-label codes for the :TYPE column,
            # and the labelId:LONG column itself can be kept or dropped later by neo4j-admin if not needed.
            # If neo4j-admin needs a specific name like 'labelId' without ':LONG' after this processing,
            # that column renaming can be added as a final step when saving this DataFrame.
        ]
    },
    "types.edges": {
        "data_filename": "types.edges.csv",
        "header_filename": "types.edges.header.csv",
        "foreign_keys": [
            { "fk_col": ":START_ID", "source_node_key": "entities", "source_id_col_in_source_file": "lid:ID"},
            { "fk_col": ":END_ID",   "source_node_key": "types",    "source_id_col_in_source_file": "lid:ID"},
            { "fk_col": "labelId:LONG", "source_node_key": "edge-labels", "source_id_col_in_source_file": "labelId:ID"}
        ]
    }
}
# --- 配置区域结束 ---

def load_headers_from_file(header_filepath):
    """从指定的 .header.csv 文件加载列名列表。"""
    try:
        with open(header_filepath, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
        if not header_line:
            print(f"  警告: 表头文件 {header_filepath} 为空。")
            return None
        column_names = [col.strip() for col in header_line.split(',')]
        if not all(c.strip() for c in column_names if isinstance(c, str)):
             print(f"  警告: 表头文件 {header_filepath} 中包含无效的空列名: {column_names}")
        return column_names
    except FileNotFoundError:
        print(f"  错误: 表头文件 {header_filepath} 未找到。")
        return None
    except Exception as e:
        print(f"  读取表头文件 {header_filepath} 时发生错误: {e}")
        return None

def generate_readme(original_headers_map, final_dataframes_info):
    """生成README文件，记录转换过程和最终表头。"""
    print(f"生成README文件到 {README_FILE}...")
    readme_content = ["# 数据处理流程与最终文件说明 (v6)\n\n"]
    readme_content.append(f"原始数据来自: `{BASE_ORIGINAL_DATA_PATH}` (数据文件无表头，表头在单独的 .header.csv 文件中)\n")
    readme_content.append(f"最终处理后文件（包含表头）位于: `{FINAL_OUTPUT_PATH}`\n\n")

    readme_content.append("## 1. 原始文件表头 (来自 .header.csv 文件)\n")
    for filename, header_list in original_headers_map.items():
        readme_content.append(f"* **{filename}**: `{','.join(header_list)}`\n")

    readme_content.append("\n## 2. 节点文件去重策略\n")
    readme_content.append(f"* **{CONFIG['entities']['data_filename']}**: 基于列 **'{CONFIG['entities']['id_col']}'** 去重。对于每个唯一ID，保留其 **'{CONFIG['entities']['score_col']}'** 列值最高的那条记录。\n")
    readme_content.append(f"* **{CONFIG['types']['data_filename']}**: 基于列 **'{CONFIG['types']['id_col']}'** 去重，保留每个ID第一次出现的记录。\n")
    readme_content.append(f"* **{CONFIG['edge-labels']['data_filename']}**: 基于列 **'{CONFIG['edge-labels']['id_col']}'** 去重，保留每个ID第一次出现的记录。\n")
    readme_content.append("* **ID前缀化**: 在此版本脚本中，**没有对节点ID进行前缀添加**，假定原始ID在 `entities`, `types`, `edge-labels` 之间已具备全局唯一性。\n")

    readme_content.append("\n## 3. 关系文件处理策略\n")
    readme_content.append(f"* **外键参照完整性过滤**: 对于 `other.edges.csv` 和 `types.edges.csv`：\n")
    readme_content.append(f"  * `:START_ID` 和 `:END_ID` (以及 `labelId:LONG`) 列的值被检查，确保它们指向在步骤2中去重后仍然存在的节点ID。\n")
    readme_content.append(f"  * 如果关系指向一个已被移除（或原始就不存在）的节点ID，该关系行将被从最终文件中移除。\n")
    readme_content.append(f"* **关系类型 (`:TYPE`) 填充**: `other.edges.csv` 和 `types.edges.csv` 中的 `:TYPE` 列，其值被替换为从（去重后的）`{CONFIG['edge-labels']['data_filename']}` 文件的 **`{CONFIG['edge-labels']['rel_type_code_col']}`** 列中查找到的对应字符串（通过 `labelId:LONG` 映射）。\n")
    readme_content.append(f"* **`labelId:LONG` 列**: 此列在关系文件中保持不变，其值是原始的数字ID，用于查找关系类型代码。\n")

    readme_content.append("\n## 4. 最终输出文件的表头 (位于 " + FINAL_OUTPUT_PATH + ")\n")
    for filename, info in final_dataframes_info.items(): # info 包含 df 和 status
        header_str = "错误: DataFrame未生成或表头不可用"
        if info.get("df") is not None and not info["df"].empty:
            header_str = ','.join(info["df"].columns.tolist())
        elif info.get("status") == "loaded_original_header_only":
            header_str = f"原始表头: {','.join(info.get('original_header',[]))} (文件处理失败或为空)"
        readme_content.append(f"* **{filename}**: `{header_str}`\n")
    
    try:
        os.makedirs(os.path.dirname(README_FILE), exist_ok=True)
        with open(README_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(readme_content))
        print(f"README 文件已成功保存到: {README_FILE}")
    except Exception as e:
        print(f"错误: 保存README文件失败: {e}")


def main_pipeline():
    print(f"开始数据处理流程 (v6 - 无ID前缀，先节点去重后关系过滤和类型填充)。输出到: {FINAL_OUTPUT_PATH}")
    if os.path.exists(FINAL_OUTPUT_PATH):
        print(f"清理已存在的输出目录: {FINAL_OUTPUT_PATH}")
        shutil.rmtree(FINAL_OUTPUT_PATH)
    os.makedirs(FINAL_OUTPUT_PATH, exist_ok=True)

    # 存储加载的DataFrames和原始表头
    dataframes = {}
    original_headers_map = {}
    # 存储去重后保留的节点ID集合
    kept_node_ids = {
        "entities": set(),
        "types": set(),
        "edge-labels": set()
    }

    # --- 阶段 1: 加载所有原始数据文件（带表头）---
    print("\n" + "="*15 + " 阶段 1: 加载所有原始数据文件 " + "="*15)
    all_file_keys = ["entities", "types", "edge-labels", "other.edges", "types.edges"]
    for key in all_file_keys:
        conf = CONFIG[key]
        data_filepath = os.path.join(BASE_ORIGINAL_DATA_PATH, conf["data_filename"])
        header_filepath = os.path.join(BASE_ORIGINAL_DATA_PATH, conf["header_filename"])
        
        print(f"  加载: {conf['data_filename']} (使用表头: {conf['header_filename']})")
        headers = load_headers_from_file(header_filepath)
        original_headers_map[conf["data_filename"]] = headers if headers else ["错误: 表头加载失败"]
        
        if not headers:
            print(f"    错误: 无法加载 {conf['header_filename']} 的表头。跳过 {conf['data_filename']}。")
            dataframes[conf["data_filename"]] = None
            continue
        try:
            # 初始加载全部为字符串，后续按需转换类型
            df = pd.read_csv(data_filepath, header=None, names=headers, dtype=str, 
                             low_memory=False, keep_default_na=False, na_values=[''])
            dataframes[conf["data_filename"]] = df
            print(f"    已加载 {len(df)} 行数据从 {conf['data_filename']}.")
        except Exception as e:
            print(f"    错误加载数据文件 {data_filepath}: {e}")
            traceback.print_exc()
            dataframes[conf["data_filename"]] = None

    # --- 阶段 2: 节点文件去重 ---
    print("\n" + "="*15 + " 阶段 2: 节点文件去重 " + "="*15)
    # (A) entities.csv 去重 (保留最高分)
    entities_conf = CONFIG["entities"]
    df_entities = dataframes.get(entities_conf["data_filename"])
    if df_entities is not None:
        id_col_e = entities_conf["id_col"]
        score_col_e = entities_conf["score_col"]
        print(f"  去重 {entities_conf['data_filename']}: 基于 '{id_col_e}'，保留 '{score_col_e}' 最高分记录...")
        try:
            # 清理和转换score列为数值类型
            df_entities[score_col_e] = pd.to_numeric(df_entities[score_col_e], errors='coerce')
            df_entities.dropna(subset=[score_col_e], inplace=True) # 移除score无法转换的行

            df_entities.sort_values(by=[id_col_e, score_col_e], ascending=[True, False], inplace=True)
            df_entities.drop_duplicates(subset=[id_col_e], keep='first', inplace=True)
            dataframes[entities_conf["data_filename"]] = df_entities
            kept_node_ids["entities"] = set(df_entities[id_col_e].astype(str).str.strip().dropna().unique())
            print(f"    {entities_conf['data_filename']} 去重后剩余 {len(df_entities)} 行。保留了 {len(kept_node_ids['entities'])} 个唯一实体ID。")
        except KeyError as ke:
            print(f"    错误: 列名配置错误导致KeyError进行实体去重: {ke}。请检查CONFIG中 '{id_col_e}' 或 '{score_col_e}' 是否正确。")
            kept_node_ids["entities"] = set()
        except Exception as e:
            print(f"    错误在 {entities_conf['data_filename']} 去重时发生: {e}")
            traceback.print_exc()
            kept_node_ids["entities"] = set()
    else:
        print(f"  跳过 {entities_conf['data_filename']} 去重，因其未成功加载。")
        kept_node_ids["entities"] = set()

    # (B) types.csv 去重 (保留第一个)
    types_conf = CONFIG["types"]
    df_types = dataframes.get(types_conf["data_filename"])
    if df_types is not None:
        id_col_t = types_conf["id_col"]
        print(f"  去重 {types_conf['data_filename']}: 基于 '{id_col_t}'，保留第一条记录...")
        try:
            original_len = len(df_types)
            df_types.drop_duplicates(subset=[id_col_t], keep='first', inplace=True)
            dataframes[types_conf["data_filename"]] = df_types
            kept_node_ids["types"] = set(df_types[id_col_t].astype(str).str.strip().dropna().unique())
            print(f"    {types_conf['data_filename']} 从 {original_len} 行去重到 {len(df_types)} 行。保留了 {len(kept_node_ids['types'])} 个唯类型ID。")
        except Exception as e:
            print(f"    错误在 {types_conf['data_filename']} 去重时发生: {e}")
            traceback.print_exc()
            kept_node_ids["types"] = set()
    else:
        print(f"  跳过 {types_conf['data_filename']} 去重，因其未成功加载。")
        kept_node_ids["types"] = set()

    # (C) edge-labels.csv 去重 (保留第一个) 并创建 labelId -> code 映射
    el_conf = CONFIG["edge-labels"]
    df_el = dataframes.get(el_conf["data_filename"])
    label_id_to_code_map = {} # 使用原始（未前缀化）的labelId作为键
    if df_el is not None:
        id_col_el = el_conf["id_col"]
        code_col_el = el_conf["rel_type_code_col"]
        print(f"  去重 {el_conf['data_filename']}: 基于 '{id_col_el}'，保留第一条记录，并创建ID到'{code_col_el}'的映射...")
        try:
            original_len = len(df_el)
            df_el.drop_duplicates(subset=[id_col_el], keep='first', inplace=True)
            dataframes[el_conf["data_filename"]] = df_el
            kept_node_ids["edge-labels"] = set(df_el[id_col_el].astype(str).str.strip().dropna().unique())
            print(f"    {el_conf['data_filename']} 从 {original_len} 行去重到 {len(df_el)} 行。保留了 {len(kept_node_ids['edge-labels'])} 个唯一关系标签ID。")
            
            if id_col_el in df_el.columns and code_col_el in df_el.columns:
                for _, row in df_el.iterrows(): # 此时df_el已经是去重后的
                    label_id_to_code_map[str(row[id_col_el]).strip()] = str(row[code_col_el]).strip()
                print(f"    已从去重后的 {el_conf['data_filename']} 创建 {len(label_id_to_code_map)} 条 '{id_col_el}' 到 '{code_col_el}' 的映射。")
            else:
                 print(f"    错误: ID列 '{id_col_el}' 或 code列 '{code_col_el}' 未在 {el_conf['data_filename']} 中找到。无法创建映射。")
        except Exception as e:
            print(f"    错误在 {el_conf['data_filename']} 去重或映射创建时发生: {e}")
            traceback.print_exc()
            kept_node_ids["edge-labels"] = set()
    else:
        print(f"  跳过 {el_conf['data_filename']} 去重和映射创建，因其未成功加载。")
        kept_node_ids["edge-labels"] = set()


    # --- STAGE 3: 关系文件外键过滤与 :TYPE 列填充 ---
    print("\n" + "="*15 + " 阶段 3: 关系文件外键过滤与 :TYPE 列填充 " + "="*15)
    for rel_key in ["other.edges", "types.edges"]:
        rel_conf = CONFIG[rel_key]
        df_rel = dataframes.get(rel_conf["data_filename"])
        if df_rel is None:
            print(f"  跳过 {rel_conf['data_filename']} 的处理，因其未成功加载。")
            continue

        print(f"  处理关系文件: {rel_conf['data_filename']}...")
        original_row_count = len(df_rel)

        # 3.1 外键过滤
        print(f"    外键过滤...")
        for fk_task in rel_conf.get("foreign_keys", []):
            fk_col = fk_task["fk_col"]
            source_node_key = fk_task["source_node_key"] # e.g., "entities", "types", "edge-labels"
            
            if fk_col not in df_rel.columns:
                print(f"      警告: 外键列 '{fk_col}' 未在 {rel_conf['data_filename']} 中找到。无法基于此列过滤。")
                continue
            if source_node_key not in kept_node_ids:
                print(f"      警告: 未找到源 '{source_node_key}' 的保留ID集合。无法基于 '{fk_col}' 过滤。")
                continue
            
            valid_ids_for_fk = kept_node_ids[source_node_key]
            if not valid_ids_for_fk: # 如果某个节点类型的保留ID集合为空
                print(f"      警告: 源 '{source_node_key}' 的保留ID集合为空。基于 '{fk_col}' 的过滤可能会移除所有行。")
            
            initial_len_before_this_fk_filter = len(df_rel)
            df_rel[fk_col] = df_rel[fk_col].astype(str).str.strip() # 清理外键列数据
            df_rel = df_rel[df_rel[fk_col].isin(valid_ids_for_fk)]
            rows_dropped_by_this_fk = initial_len_before_this_fk_filter - len(df_rel)
            if rows_dropped_by_this_fk > 0:
                print(f"      因列 '{fk_col}' 的值不在保留的 '{source_node_key}' ID集合中，移除了 {rows_dropped_by_this_fk} 行。")
        
        print(f"    外键过滤后剩余 {len(df_rel)} 行 (原 {original_row_count} 行)。")

        # 3.2 :TYPE 列填充
        if not label_id_to_code_map:
            print(f"    警告: labelId到code的映射表为空，无法为 {rel_conf['data_filename']} 填充 :TYPE 列。")
        else:
            print(f"    填充 :TYPE 列...")
            # 找到引用 edge-labels ID 的原始列名 (通常是 labelId:LONG)
            el_fk_config = next((fc for fc in rel_conf.get("foreign_keys", []) if fc["source_node_key"] == "edge-labels"), None)
            if not el_fk_config:
                print(f"      警告: 在 {rel_conf['data_filename']} 的配置中未找到引用 'edge-labels' 的外键。无法填充 :TYPE 列。")
            else:
                label_id_ref_col_in_rel = el_fk_config["fk_col"] # 这是原始的列名，如 'labelId:LONG'
                if label_id_ref_col_in_rel not in df_rel.columns:
                    print(f"      错误: 用于查找关系类型的列 '{label_id_ref_col_in_rel}' 未在 {rel_conf['data_filename']} 的当前表头中找到。")
                else:
                    unmapped_for_type = []
                    def map_to_code(original_el_id_val):
                        s_val = str(original_el_id_val).strip()
                        if pd.notna(original_el_id_val) and s_val != "":
                            code = label_id_to_code_map.get(s_val)
                            if code is None:
                                unmapped_for_type.append(s_val)
                                return "UNKNOWN_REL_TYPE_CODE" # 或者保留原:TYPE值，如果原:TYPE列存在且有意义
                            return code
                        return "EMPTY_REL_ID_FOR_TYPE" # 如果原始ID为空

                    if ":TYPE" not in df_rel.columns: # 确保:TYPE列存在
                        df_rel[":TYPE"] = "DEFAULT_REL_TYPE" # 或其他合适的默认值
                        print(f"      在 {rel_conf['data_filename']} 中创建了新的 :TYPE 列。")
                    
                    df_rel[":TYPE"] = df_rel[label_id_ref_col_in_rel].apply(map_to_code)
                    if unmapped_for_type:
                        unique_unmapped_codes = sorted(list(set(unmapped_for_type)))
                        print(f"      警告: 为填充 :TYPE 列时，{len(unique_unmapped_codes)} 个唯一的 '{label_id_ref_col_in_rel}' 值在edge-labels映射中未找到对应code (例: {unique_unmapped_codes[:5]})。:TYPE被设为 'UNKNOWN_REL_TYPE_CODE'。")
                    print(f"      :TYPE 列已使用 {CONFIG['edge-labels']['rel_type_code_col']} 列的值填充。")
        
        dataframes[rel_conf["data_filename"]] = df_rel # 更新处理后的DataFrame

    # --- STAGE 4: 保存所有最终的DataFrames ---
    print("\n" + "="*15 + " 阶段 4: 保存所有最终文件 " + "="*15)
    final_df_info_for_readme = {} # 用于生成README时获取最终表头
    for filename, df_final in dataframes.items(): # filename是原始文件名，如 "entities.csv"
        if df_final is not None:
            output_filepath = os.path.join(FINAL_OUTPUT_PATH, filename)
            try:
                df_final.to_csv(output_filepath, index=False, header=True) # 写入表头
                print(f"  已保存最终文件: {output_filepath} (共 {len(df_final)} 行)")
                final_df_info_for_readme[filename] = {"df": df_final, "status": "saved"}
            except Exception as e:
                print(f"  错误保存文件 {output_filepath}: {e}")
                traceback.print_exc()
                final_df_info_for_readme[filename] = {"df": None, "status": "save_failed"}
        else:
            print(f"  文件 {filename} 未处理或处理失败，不进行保存。")
            # 尝试从原始表头获取信息给README
            original_header = original_headers_map.get(filename, ["未知表头"])
            final_df_info_for_readme[filename] = {"df": None, "status": "processing_failed", "original_header": original_header}


    # --- STAGE 5: 生成README文件 ---
    print("\n" + "="*15 + " 阶段 5: 生成README文件 " + "="*15)
    generate_readme(original_headers_map, final_df_info_for_readme)

    print("\n--- 所有处理阶段完毕 ---")
    print(f"请检查 '{FINAL_OUTPUT_PATH}' 目录下的新文件以及 '{README_FILE}'。")
    print("运行 neo4j-admin import 时，请确保使用这些新生成的、包含表头的文件。")

if __name__ == "__main__":
    main_pipeline()