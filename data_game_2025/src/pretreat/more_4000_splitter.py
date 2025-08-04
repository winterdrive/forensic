'''
Winston 從 datagame_sms_stage1_raw_TEXT_ONLY.csv 題目再撈出id 8164之後的 4000 條筆
output為more_4000.csv
輸出檔案的格式如下範例：

sms_id,sms_body
sms_id,sms_body
1,您好，這是中國信託銀行的通知，您的貸款款項尚未繳納，請儘速償還以維護您的信用。如已繳款，請忽略此訊息。如有問題，請致電客服專線。
...

放在 data_game_2025/data/raw 內
'''

import csv

def get_excluded_ids(filenames):
    excluded_ids = set()
    for fname in filenames:
        with open(fname, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sms_id = row.get('sms_id')
                if sms_id is not None:
                    excluded_ids.add(sms_id)
    return excluded_ids

def main():
    # Input and output paths
    input_file = '/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv'
    output_file = 'data_game_2025/data/raw/more_4000.csv'
    start_id = 8164
    num_rows = 4000
    exclude_files = [
        'data_game_2025/data/raw/name_1000.csv',
        'data_game_2025/data/raw/raw_4000.csv',
        'data_game_2025/data/raw/travel_1000.csv',
    ]

    # Get excluded ids
    excluded_ids = get_excluded_ids(exclude_files)

    # Read all rows from the input CSV
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Filter rows with sms_id > 8164, not in excluded_ids, and get the next 4000
    filtered_rows = [row for row in rows if int(row.get('sms_id', 0)) > start_id and row.get('sms_id') not in excluded_ids]
    selected_rows = filtered_rows[:num_rows]

    # Write to output CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['sms_id', 'sms_body'])
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({
                'sms_id': row.get('sms_id', ''),
                'sms_body': row.get('sms_body', '')
            })

if __name__ == '__main__':
    main()