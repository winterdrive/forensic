#!/usr/bin/env python3
"""
将只包含 sms_id 的结果文件与包含完整簡訊内容的源文件进行 join，
输出格式为 sms_id,sms_body

使用方法:
python src/pretreat/join_sms_body.py --input <result_file.csv> --source <source_file.csv> --output <output_file.csv>

例如:
python src/pretreat/join_sms_body.py \
    --input data/results/labled/stage2/測試賽上傳/name_result_0802晚.csv \
    --source data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv \
    --output data/results/labled/stage2/測試賽上傳/name_result_0802晚_with_body.csv
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def join_sms_body(input_file: str, source_file: str, output_file: str) -> None:
    """
    将只包含 sms_id 的结果文件与包含完整簡訊内容的源文件进行 join
    
    Args:
        input_file: 只包含 sms_id 的输入文件路径
        source_file: 包含完整 sms_id,sms_body 的源文件路径  
        output_file: 输出文件路径
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"源文件不存在: {source_file}")
    
    print(f"正在读取输入文件: {input_file}")
    # 读取只包含 sms_id 的结果文件
    try:
        result_df = pd.read_csv(input_file)
        print(f"输入文件包含 {len(result_df)} 条记录")
    except Exception as e:
        raise Exception(f"读取输入文件失败: {e}")
    
    # 检查输入文件是否包含 sms_id 列
    if 'sms_id' not in result_df.columns:
        raise ValueError(f"输入文件缺少 sms_id 列。当前列: {result_df.columns.tolist()}")
    
    print(f"正在读取源文件: {source_file}")
    # 读取包含完整内容的源文件
    try:
        source_df = pd.read_csv(source_file)
        print(f"源文件包含 {len(source_df)} 条记录")
    except Exception as e:
        raise Exception(f"读取源文件失败: {e}")
    
    # 检查源文件是否包含必要的列
    required_columns = ['sms_id', 'sms_body']
    missing_columns = [col for col in required_columns if col not in source_df.columns]
    if missing_columns:
        raise ValueError(f"源文件缺少必要列: {missing_columns}。当前列: {source_df.columns.tolist()}")
    
    print("正在进行数据 join...")
    # 进行 join 操作
    joined_df = result_df.merge(
        source_df[['sms_id', 'sms_body']], 
        on='sms_id', 
        how='left'
    )
    
    # 检查是否有未匹配的记录
    missing_count = joined_df['sms_body'].isna().sum()
    if missing_count > 0:
        print(f"警告: 有 {missing_count} 条记录在源文件中找不到对应的簡訊内容")
        missing_ids = joined_df[joined_df['sms_body'].isna()]['sms_id'].tolist()
        print(f"缺失的 sms_id: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")
    
    # 只保留 sms_id 和 sms_body 列
    output_df = joined_df[['sms_id', 'sms_body']].copy()
    
    # 创建输出目录
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在保存到: {output_file}")
    # 保存结果
    try:
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"成功保存 {len(output_df)} 条记录到 {output_file}")
    except Exception as e:
        raise Exception(f"保存文件失败: {e}")
    
    # 显示统计信息
    print("\n=== 处理统计 ===")
    print(f"输入记录数: {len(result_df)}")
    print(f"成功匹配记录数: {len(output_df) - missing_count}")
    print(f"未匹配记录数: {missing_count}")
    print(f"输出记录数: {len(output_df)}")


def main():
    parser = argparse.ArgumentParser(
        description="将只包含 sms_id 的结果文件与包含完整簡訊内容的源文件进行 join",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例子:
  python src/pretreat/join_sms_body.py \\
      --input data/results/labled/stage2/測試賽上傳/name_result_0802晚.csv \\
      --source data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv \\
      --output data/results/labled/stage2/測試賽上傳/name_result_0802晚_with_body.csv

  python src/pretreat/join_sms_body.py \\
      --input data/results/labled/stage2/測試賽上傳/category_result_0802晚.csv \\
      --source data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv \\
      --output data/results/labled/stage2/測試賽上傳/category_result_0802晚_with_body.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='只包含 sms_id 的输入文件路径'
    )
    
    parser.add_argument(
        '--source', '-s', 
        required=True,
        help='包含完整 sms_id,sms_body 的源文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True, 
        help='输出文件路径'
    )
    
    args = parser.parse_args()
    
    try:
        join_sms_body(args.input, args.source, args.output)
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
