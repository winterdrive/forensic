#!/usr/bin/env python3
"""
批量处理多个只包含 sms_id 的结果文件，为它们添加簡訊内容

使用方法:
python src/pretreat/batch_join_sms_body.py --input-dir <input_directory> --source <source_file.csv> --output-dir <output_directory>

例如:
python src/pretreat/batch_join_sms_body.py \
    --input-dir data/results/labled/stage2/測試賽上傳 \
    --source data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv \
    --output-dir data/results/labled/stage2/測試賽上傳/with_body
"""

import pandas as pd
import argparse
import os
from pathlib import Path
import glob


def batch_join_sms_body(input_dir: str, source_file: str, output_dir: str, pattern: str = "*.csv") -> None:
    """
    批量处理多个只包含 sms_id 的结果文件，为它们添加簡訊内容
    
    Args:
        input_dir: 输入文件目录
        source_file: 包含完整 sms_id,sms_body 的源文件路径
        output_dir: 输出文件目录
        pattern: 文件匹配模式，默认为 "*.csv"
    """
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 检查源文件是否存在
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"源文件不存在: {source_file}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 查找所有匹配的文件
    input_pattern = os.path.join(input_dir, pattern)
    input_files = glob.glob(input_pattern)
    
    if not input_files:
        print(f"在目录 {input_dir} 中没有找到匹配模式 {pattern} 的文件")
        return
    
    print(f"找到 {len(input_files)} 个文件待处理")
    print(f"正在读取源文件: {source_file}")
    
    # 一次性读取源文件
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
    
    # 处理每个文件
    processed_count = 0
    total_input_records = 0
    total_output_records = 0
    total_missing_records = 0
    
    for input_file in input_files:
        try:
            print(f"\n正在处理: {input_file}")
            
            # 读取输入文件
            result_df = pd.read_csv(input_file)
            input_records = len(result_df)
            total_input_records += input_records
            print(f"  输入记录数: {input_records}")
            
            # 检查输入文件是否包含 sms_id 列
            if 'sms_id' not in result_df.columns:
                print(f"  跳过文件: 缺少 sms_id 列。当前列: {result_df.columns.tolist()}")
                continue
            
            # 进行 join 操作
            joined_df = result_df.merge(
                source_df[['sms_id', 'sms_body']], 
                on='sms_id', 
                how='left'
            )
            
            # 检查是否有未匹配的记录
            missing_count = joined_df['sms_body'].isna().sum()
            total_missing_records += missing_count
            if missing_count > 0:
                print(f"  警告: 有 {missing_count} 条记录在源文件中找不到对应的簡訊内容")
            
            # 只保留 sms_id 和 sms_body 列
            output_df = joined_df[['sms_id', 'sms_body']].copy()
            
            # 生成输出文件名
            input_filename = Path(input_file).name
            output_filename = input_filename.replace('.csv', '_with_body.csv')
            output_file = os.path.join(output_dir, output_filename)
            
            # 保存结果
            output_df.to_csv(output_file, index=False, encoding='utf-8')
            output_records = len(output_df)
            total_output_records += output_records
            print(f"  成功保存 {output_records} 条记录到 {output_file}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  处理文件 {input_file} 时出错: {e}")
            continue
    
    # 显示总体统计信息
    print(f"\n=== 批量处理统计 ===")
    print(f"找到文件数: {len(input_files)}")
    print(f"成功处理文件数: {processed_count}")
    print(f"总输入记录数: {total_input_records}")
    print(f"总成功匹配记录数: {total_output_records - total_missing_records}")
    print(f"总未匹配记录数: {total_missing_records}")
    print(f"总输出记录数: {total_output_records}")


def main():
    parser = argparse.ArgumentParser(
        description="批量处理多个只包含 sms_id 的结果文件，为它们添加簡訊内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例子:
  # 处理目录下所有 CSV 文件
  python src/pretreat/batch_join_sms_body.py \\
      --input-dir data/results/labled/stage2/測試賽上傳 \\
      --source data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv \\
      --output-dir data/results/labled/stage2/測試賽上傳/with_body

  # 只处理特定模式的文件
  python src/pretreat/batch_join_sms_body.py \\
      --input-dir data/results/labled/stage2/測試賽上傳 \\
      --source data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv \\
      --output-dir data/results/labled/stage2/測試賽上傳/with_body \\
      --pattern "*_result_*.csv"
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='包含只有 sms_id 文件的输入目录'
    )
    
    parser.add_argument(
        '--source', '-s', 
        required=True,
        help='包含完整 sms_id,sms_body 的源文件路径'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        required=True, 
        help='输出文件目录'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        default='*.csv',
        help='文件匹配模式，默认为 "*.csv"'
    )
    
    args = parser.parse_args()
    
    try:
        batch_join_sms_body(args.input_dir, args.source, args.output_dir, args.pattern)
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
