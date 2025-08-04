'''
Winston 從 datagame_sms_stage1_raw_TEXT_ONLY.csv 題目均分給四個人，
(Winston)(Renhe)(Pikas)(JC)
output為四個檔案，檔名後綴標注是誰的，bonus_winston.csv等
輸出檔案的格式為：
sms_id,sms_body, category
放在 data_game_2025/data/raw 內
'''

import csv
import os

def main():
    # Input and output paths
    input_file = '/Users/winstontang/PycharmProjects/forensic/data_game_2025/data/raw/datagame_sms_stage1_raw_TEXT_ONLY.csv'
    output_dir = 'data_game_2025/data/raw'
    people = ['winston', 'renhe', 'pikas', 'jc']

    # Read all rows from the input CSV
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Split rows evenly among the four people
    split_size = len(rows) // len(people)
    remainder = len(rows) % len(people)

    splits = []
    start = 0
    for i in range(len(people)):
        end = start + split_size + (1 if i < remainder else 0)
        splits.append(rows[start:end])
        start = end

    # Write each split to a separate CSV file
    for person, split_rows in zip(people, splits):
        output_path = os.path.join(output_dir, f'bonus_profile_{person}.csv')
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['sms_id', 'sms_body', 'category'])
            writer.writeheader()
            for row in split_rows:
                writer.writerow({
                    'sms_id': row.get('sms_id', ''),
                    'sms_body': row.get('sms_body', ''),
                    'category': row.get('category', '')
                })

if __name__ == '__main__':
    main()