import csv
import sys

# 用法：python csv_sorter.py input.csv output.csv

def sort_csv(input_path, output_path):
    """
    讀取 input_path 的 CSV，根據 id 欄位排序，並寫入 output_path。
    支援多種 id 欄位名稱：'id', 'sms_id'。
    """
    with open(input_path, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        fieldnames = reader.fieldnames
        
        # 找到 id 欄位名稱
        id_field = None
        for field in ['sms_id', 'id']:
            if field in fieldnames:
                id_field = field
                break
        
        if not id_field:
            raise ValueError(f"找不到 id 欄位，可用欄位: {fieldnames}")
        
        # 依照 id 欄位（轉為 int）排序
        rows.sort(key=lambda x: int(x[id_field]))

    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('用法: python csv_sorter.py <input.csv> [output.csv]')
        sys.exit(1)
    input_path = sys.argv[1]
    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        if input_path.lower().endswith('.csv'):
            output_path = input_path[:-4] + '_sorted.csv'
        else:
            output_path = input_path + '_sorted.csv'
    sort_csv(input_path, output_path)
