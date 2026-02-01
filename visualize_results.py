import json
import os
import shutil
import argparse
from tqdm import tqdm

def save_matches_to_folder(matches, target_folder):
    """
    辅助函数：将匹配列表中的图片复制到指定文件夹
    """
    os.makedirs(target_folder, exist_ok=True)
    
    for match in matches:
        rank = match['rank']
        is_correct = match['is_correct']
        src_img_path = match['img_path']
        
        # 标记正确与否的前缀
        status = "CORRECT" if is_correct else "WRONG"
        
        # 构造新文件名: rank_{排名}_{状态}_{原文件名}
        filename = os.path.basename(src_img_path)
        new_filename = f"rank_{rank:02d}_{status}_{filename}"
        dst_img_path = os.path.join(target_folder, new_filename)

        try:
            shutil.copy(src_img_path, dst_img_path)
        except FileNotFoundError:
            print(f"[Warning] Image not found: {src_img_path}")
        except Exception as e:
            print(f"[Error] Copy failed for {src_img_path}: {e}")

def visualize_compare(json_path1, json_path2, output_dir, max_queries=50):
    """
    Args:
        json_path1: 主模型结果 (用于筛选条件：Top1必须对，且总数优于json2)
        json_path2: 对比模型结果
        output_dir: 结果输出的根目录
        max_queries: 限制可视化的 query 数量
    """
    if not os.path.exists(json_path1) or not os.path.exists(json_path2):
        print("Error: One of the JSON files was not found.")
        return

    model_name_1 = os.path.splitext(os.path.basename(json_path1))[0]
    model_name_2 = os.path.splitext(os.path.basename(json_path2))[0]
    
    if model_name_1 == model_name_2:
        model_name_1 += "_1"
        model_name_2 += "_2"

    print(f"Loading Model 1 (Base/Filter): {model_name_1}...")
    with open(json_path1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    print(f"Loading Model 2 (Reference): {model_name_2}...")
    with open(json_path2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # 建立映射方便查找
    data2_map = {item['query_id']: item for item in data2}

    os.makedirs(output_dir, exist_ok=True)
    print(f"Start processing. Output to: {output_dir}")
    print("Applying filters: Model 1 Top-1 must be correct AND Model 1 count >= Model 2 count.")

    count = 0
    # 遍历数据
    for item1 in tqdm(data1):
        # 达到数量限制则停止
        if max_queries is not None and count >= max_queries:
            break
            
        query_id = item1['query_id']
        matches1 = item1['matches']
        
        # 确保 Model 2 中也有这个 query
        if query_id not in data2_map:
            continue
        
        item2 = data2_map[query_id]
        matches2 = item2['matches']

        # ==========================================
        # 新增筛选逻辑 Start
        # ==========================================
        
        # 1. 检查条件一：Model 1 的 Top-1 必须是对的
        # 假设 matches 是按 rank 排序的，取第一个
        if not matches1 or not matches1[0]['is_correct']:
            continue # Top-1 错了，跳过

        # 2. 检查条件二：Model 1 的正确总数 >= Model 2 的正确总数
        correct_count_1 = sum(1 for m in matches1 if m['is_correct'])
        correct_count_2 = sum(1 for m in matches2 if m['is_correct'])

        if correct_count_1 < correct_count_2+1:
            continue # Model 1 表现不如 Model 2，跳过

        # ==========================================
        # 新增筛选逻辑 End
        # ==========================================

        # 如果通过了筛选，开始生成可视化文件夹
        query_text = item1['query_text']
        safe_text = "".join([c for c in query_text if c.isalnum() or c in (' ', '_')]).strip()[:30]
        query_folder_name = f"q{query_id}_{safe_text}"
        query_root_path = os.path.join(output_dir, query_folder_name)
        os.makedirs(query_root_path, exist_ok=True)

        # 保存 Query 文本
        with open(os.path.join(query_root_path, "query_text.txt"), "w", encoding='utf-8') as f:
            f.write(f"Query ID: {query_id}\\n")
            f.write(f"Query PID: {item1['query_pid']}\\n")
            f.write(f"Stats Model 1 ({model_name_1}): Top-1 Correct, Total Correct: {correct_count_1}\\n")
            f.write(f"Stats Model 2 ({model_name_2}): Total Correct: {correct_count_2}\\n")
            f.write("-" * 20 + "\\n")
            f.write(query_text)

        # 保存两个模型的图片
        path1 = os.path.join(query_root_path, model_name_1)
        save_matches_to_folder(matches1, path1)
        
        path2 = os.path.join(query_root_path, model_name_2)
        save_matches_to_folder(matches2, path2)

        count += 1

    print(f"Visualization generation complete. Found {count} queries satisfying the conditions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json1", type=str, required=True, help="Main model json (Filters apply to this)")
    parser.add_argument("--json2", type=str, required=True, help="Reference model json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max_num", type=int, default=50, help="Max number of visualizations")
    
    args = parser.parse_args()
    
    visualize_compare(args.json1, args.json2, args.output_dir, args.max_num)

# python visualize_results.py --json1 ./epoch_30_top10_results_better.json --json2 ./epoch_30_top10_results.json --output_dir ./pic/res_img_cuhk --max_num 50