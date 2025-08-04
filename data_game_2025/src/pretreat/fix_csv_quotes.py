import sys
import os

def fix_csv_by_header(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 取得 header 與欄位數
    header = lines[0].rstrip('\n').rstrip('\r')
    columns = [col.strip() for col in header.split(',')]
    col_count = len(columns)

    # 強制檢查前兩欄必為 sms_id, sms_body
    if col_count < 2 or columns[0].lower() != 'sms_id' or columns[1].lower() != 'sms_body':
        print(f"錯誤：header 前兩欄必須為 sms_id, sms_body，實際為：{columns[:2]}")
        sys.exit(1)

    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write(header + '\n')
        for line in lines[1:]:
            line = line.rstrip('\n').rstrip('\r')
            if not line.strip():
                continue
            parts = line.split(',')
            # 若欄位數正確，直接寫出
            if len(parts) == col_count:
                row = [p.strip() for p in parts]
            # 若欄位數多於 header，合併多出來的欄位到 sms_body（第二欄）
            elif len(parts) > col_count:
                row = [p.strip() for p in parts[:1]]  # sms_id
                # 合併所有 sms_body 相關欄位
                body = ','.join(parts[1:]).strip()
                # 強制加上雙引號，並 escape 內部雙引號
                body = f'"{body.replace("\"", "\"\"")}"'
                row.append(body)
            else:
                # 欄位數不足，直接寫出（或可加警告）
                row = [p.strip() for p in parts]
            outfile.write(','.join(row) + '\n')
    print(f"已產生修正後的 CSV：{output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python fix_csv_by_header.py <input.csv> [output.csv]")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_csv)[0] + "_fixed.csv"
    fix_csv_by_header(input_csv, output_csv)