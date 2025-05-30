import csv
import argparse
import logging
import sys # 需要导入 sys 模块来访问原始参数
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable, AuthError

# ... (保留脚本顶部的 label_to_id_field_map, logger, setup_logging, 
# parse_node_header, parse_relationship_header, 
# import_nodes_tx, import_relationships_tx 函数定义不变) ...

def main():
    # argparse 配置：只定义脚本直接认识的参数
    parser = argparse.ArgumentParser(
        description="通过Bolt连接从CSV文件导入数据到Neo4j，模拟neo4j-admin import。",
        allow_abbrev=False # 禁止缩写参数名，增加明确性
    )
    parser.add_argument('--uri', default="bolt://localhost:7687", help="Neo4j Bolt URI。 (默认: bolt://localhost:7687)")
    parser.add_argument('--user', default="neo4j", help="Neo4j 用户名。 (默认: neo4j)")
    parser.add_argument('--password', required=True, help="Neo4j 密码。")
    parser.add_argument('--database', default="neo4j", help="要导入数据的Neo4j数据库名称。 (默认: neo4j)")
    parser.add_argument('--report-file', default="import-log.log", help="报告日志文件的路径。 (默认: import-log.log)")
    
    # --relationships 参数定义保持不变
    parser.add_argument('--relationships', type=str, action='append', dest='relationships_definitions_raw', 
                        help="关系定义，格式为 'header.csv,data.csv'。可多次指定。")

    # 解析已知的参数，未知的参数会进入 unknown_args 列表
    # 注意：我们从 sys.argv[1:] 中解析，因为 sys.argv[0] 是脚本名
    known_args, unknown_args = parser.parse_known_args(args=None if sys.argv[1:] else ['--help'])


    # --- 手动处理 --nodes:LabelName=value 参数 ---
    nodes_definitions = []
    remaining_unknown = [] # 存储真正无法识别的参数

    idx = 0
    while idx < len(unknown_args):
        arg = unknown_args[idx]
        if arg.startswith("--nodes:"):
            # 期望格式 --nodes:Label=header.csv,data.csv
            parts = arg.split('=', 1)
            if len(parts) == 2:
                option_string = parts[0] # 例如 --nodes:Entity
                value_string = parts[1]  # 例如 /path/header.csv,/path/data.csv
                
                label = option_string.split(":")[-1]
                try:
                    header_file, data_file = [v.strip() for v in value_string.split(',')]
                    if not header_file or not data_file: # 确保分割后两部分都存在
                        raise ValueError("路径不能为空")
                except ValueError as e:
                    parser.error(f"参数 {option_string} 的值 '{value_string}' 格式错误。"
                                 f"期望格式 'header.csv,data.csv'。错误: {e}")
                
                nodes_definitions.append({
                    "label": label, "header": header_file, "data": data_file
                })
            else: # 格式不正确，例如 --nodes:Entity 没有值或没有 '='
                remaining_unknown.append(arg)
        else:
            remaining_unknown.append(arg)
        idx += 1

    if remaining_unknown:
        parser.error(f"存在无法识别的参数: {', '.join(remaining_unknown)}")

    # 将手动解析的 nodes_definitions 添加到 known_args 命名空间中
    known_args.nodes_definitions = nodes_definitions
    # --- 手动处理结束 ---


    # 处理原始的 --relationships 参数值 (如果存在)
    processed_relationships_definitions = []
    if known_args.relationships_definitions_raw:
        for item in known_args.relationships_definitions_raw:
            try:
                header_file, data_file = [v.strip() for v in item.split(',')]
                processed_relationships_definitions.append({"header": header_file, "data": data_file})
            except ValueError:
                parser.error(f"参数 --relationships 的值 '{item}' 格式错误。期望格式 'header.csv,data.csv'")
    known_args.relationships_definitions = processed_relationships_definitions # 使用处理后的列表


    setup_logging(known_args.report_file)
    logger.info(f"启动导入过程，参数 (处理后): {vars(known_args)}") # vars(known_args) 更易读

    driver = None
    try:
        driver = GraphDatabase.driver(known_args.uri, auth=basic_auth(known_args.user, known_args.password))
        driver.verify_connectivity()
        logger.info(f"成功连接到 Neo4j: {known_args.uri}")

        # 节点导入 (使用 known_args.nodes_definitions)
        if hasattr(known_args, 'nodes_definitions') and known_args.nodes_definitions:
            for node_def in known_args.nodes_definitions:
                label = node_def["label"]
                header_path = node_def["header"]
                data_path = node_def["data"]
                logger.info(f"处理节点定义: Label={label}, Header={header_path}, Data={data_path}")

                try:
                    with open(header_path, 'r', encoding='utf-8') as hf:
                        header_line = hf.readline().strip()
                    
                    all_property_keys_from_header, id_field_details = parse_node_header(header_line)
                    
                    if not id_field_details or not id_field_details.get('name'):
                        logger.error(f"跳过节点导入 {label}: 在头文件 {header_path} 中未找到ID字段 (例如 'fieldName:ID')。")
                        continue
                    
                    id_field_name = id_field_details['name']
                    id_field_label_hint = id_field_details['label_hint']

                    if id_field_label_hint and id_field_label_hint != label:
                        logger.warning(f"参数指定的标签 '{label}' 与头文件 {header_path} 中的ID标签提示 '{id_field_label_hint}' 不同。将使用 '{label}' 作为主标签，'{id_field_name}' 作为ID属性。")

                    # 使用全局字典存储标签和ID字段的映射
                    global label_to_id_field_map 
                    label_to_id_field_map[label] = id_field_name
                    
                    with open(data_path, 'r', encoding='utf-8') as f_data:
                        reader = csv.DictReader(f_data, fieldnames=all_property_keys_from_header)
                        rows_to_import = list(reader)

                    if not rows_to_import:
                        logger.info(f"在 {data_path} 中没有找到标签为 {label} 的数据。")
                        continue
                    
                    logger.info(f"为标签 {label} 导入 {len(rows_to_import)} 个节点, ID字段: {id_field_name}, CSV属性: {all_property_keys_from_header}")
                    with driver.session(database=known_args.database) as session:
                        count = session.execute_write(import_nodes_tx, label, id_field_name, rows_to_import)
                        logger.info(f"已导入/合并 {count} 个标签为 {label} 的节点。")

                except FileNotFoundError:
                    logger.error(f"未找到节点标签 {label} 的头文件或数据文件。搜索路径: {header_path}, {data_path}")
                except Exception as e:
                    logger.error(f"处理节点定义 {label} 失败: {e}", exc_info=True)
        else:
            logger.info("没有提供节点定义 (例如 --nodes:Entity=h.csv,d.csv)。")


        # 关系导入 (使用 known_args.relationships_definitions)
        if known_args.relationships_definitions:
            logger.info("关系导入需要APOC插件 (apoc.merge.relationship) 来处理动态类型和属性。")
            for rel_def in known_args.relationships_definitions:
                header_path = rel_def["header"]
                data_path = rel_def["data"]
                logger.info(f"处理关系定义: Header={header_path}, Data={data_path}")

                try:
                    with open(header_path, 'r', encoding='utf-8') as hf:
                        header_line = hf.readline().strip()
                    
                    parsed_rel_header_info = parse_relationship_header(header_line)
                    
                    # 使用全局字典获取ID属性
                    global label_to_id_field_map
                    start_node_actual_id_prop = label_to_id_field_map.get(parsed_rel_header_info['start_node_label_hint'])
                    end_node_actual_id_prop = label_to_id_field_map.get(parsed_rel_header_info['end_node_label_hint'])

                    if not start_node_actual_id_prop:
                        logger.error(f"无法确定起始节点标签 '{parsed_rel_header_info['start_node_label_hint']}' (来自 {header_path}) 的ID属性。请确保此标签已在 --nodes 中定义并具有 :ID 属性。")
                        continue
                    if not end_node_actual_id_prop:
                        logger.error(f"无法确定结束节点标签 '{parsed_rel_header_info['end_node_label_hint']}' (来自 {header_path}) 的ID属性。请确保此标签已在 --nodes 中定义并具有 :ID 属性。")
                        continue

                    full_rel_import_config = {
                        'start_id_key': parsed_rel_header_info['start_id_csv_col'],
                        'end_id_key': parsed_rel_header_info['end_id_csv_col'],
                        'type_key': parsed_rel_header_info['type_csv_col'],
                        'prop_keys_for_cypher': parsed_rel_header_info['props_csv_cols'],
                        'start_node_label_hint': parsed_rel_header_info['start_node_label_hint'],
                        'end_node_label_hint': parsed_rel_header_info['end_node_label_hint'],
                        'start_node_id_prop_name': start_node_actual_id_prop,
                        'end_node_id_prop_name': end_node_actual_id_prop,
                        'all_csv_cols_ordered': parsed_rel_header_info['all_csv_cols_ordered']
                    }
                    
                    with open(data_path, 'r', encoding='utf-8') as f_data:
                        reader = csv.DictReader(f_data, fieldnames=full_rel_import_config['all_csv_cols_ordered'])
                        rows_to_import = list(reader)
                    
                    if not rows_to_import:
                        logger.info(f"在 {data_path} 中没有找到关系数据。")
                        continue

                    logger.info(f"从 {data_path} 导入 {len(rows_to_import)} 个关系，连接 {full_rel_import_config['start_node_label_hint']} (通过 {full_rel_import_config['start_node_id_prop_name']}) 到 {full_rel_import_config['end_node_label_hint']} (通过 {full_rel_import_config['end_node_id_prop_name']})。")
                    
                    with driver.session(database=known_args.database) as session:
                        count = session.execute_write(import_relationships_tx, full_rel_import_config, rows_to_import)
                        logger.info(f"已使用APOC从 {data_path} 处理/合并 {count} 个关系。")

                except FileNotFoundError:
                    logger.error(f"未找到关系的头文件或数据文件。搜索路径: {header_path}, {data_path}")
                except ValueError as ve: 
                     logger.error(f"解析关系头文件 {header_path} 错误: {ve}", exc_info=True)
                except Exception as e:
                    if "apoc.merge.relationship" in str(e) and ("Unknown procedure" in str(e) or "NoSuchProcedureException" in str(e) or "SyntaxError" in str(e)):
                         logger.error("APOC错误: 未找到 'apoc.merge.relationship' 过程或不允许执行。 "
                                   "请确保APOC插件已正确安装在Neo4j服务器的plugins目录中，并在neo4j.conf中配置 (例如, dbms.security.procedures.unrestricted=apoc.* 或 dbms.security.procedures.allowlist=apoc.coll.*,apoc.load.*,apoc.merge.*)。", exc_info=False)
                         logger.debug("详细APOC错误: ", exc_info=True)
                    else:
                        logger.error(f"处理关系定义 {data_path} 失败: {e}", exc_info=True)
        else:
            logger.info("没有提供关系定义 (--relationships)。")
            
    except ServiceUnavailable:
        logger.error(f"无法连接到 Neo4j 服务: {known_args.uri}。请确保 Neo4j 服务器正在运行并且 Bolt 连接已启用。")
    except AuthError:
        logger.error(f"Neo4j 认证失败，用户: {known_args.user}。请检查用户名和密码。")
    except Exception as e:
        logger.error(f"发生未处理的错误: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()
            logger.info("Neo4j 驱动已关闭。")
        logger.info("导入过程结束。")

if __name__ == '__main__':
    # 确保全局字典在此处可用
    label_to_id_field_map = {} 
    main()