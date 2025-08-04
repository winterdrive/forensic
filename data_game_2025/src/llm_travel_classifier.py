#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅遊行程與服務通知分類器

透過 LLM + XML 解析器 + CSV 儲存器，判斷簡訊是否屬於旅遊行程與服務通知
並將分類結果存至指定檔案格式：travel_{model_name}_{yyyyMMdd_hhmm}.csv

作者: Forensic Data Game 2025 Team
日期: 2025/07/10
"""
import sys
import time
import traceback
import csv
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from src.utils import load_input_csv, parse_xml_response, estimate_token_count, escape_xml, sanitize_llm_xml, get_existing_classified_ids


class TravelClassifier:
    """旅遊行程與服務通知分類器"""

    def __init__(self, model_name: str = "mistralai/magistral-small",
                 base_url: str = "http://127.0.0.1:1234/v1",
                 api_key: str = None):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.prompt_content = self._load_prompt()
        self.client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

    def _load_prompt(self) -> str:
        prompt_paths = [
            Path(__file__).parent.parent / "prompt" /
            "travel_classifier_prompt.md",
            Path(__file__).parent / "travel_classifier_prompt.md"
        ]
        for prompt_path in prompt_paths:
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    return f.read()
        raise FileNotFoundError(
            f"找不到提示詞檔案，已嘗試路徑: {[str(p) for p in prompt_paths]}")

    def _create_messages_xml(self, messages: List[Dict[str, str]]) -> str:
        xml_content = []
        xml_content.append("請依據以下簡訊進行分類：\n")
        xml_content.append("<messages>")
        for msg in messages:
            escaped_content = escape_xml(msg["message"])
            xml_content.append(f'  <message id="{msg["id"]}">')
            xml_content.append(f'    <content>{escaped_content}</content>')
            xml_content.append('  </message>')
        xml_content.append("</messages>")
        return "\n".join(xml_content)

    def _call_llm(self, user_content: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.prompt_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=7196,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM API 呼叫失敗: {e}")
            return self._call_llm_direct(user_content)

    def _call_llm_direct(self, user_content: str) -> str:
        try:
            url = f"{self.base_url}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.prompt_content},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.1,
                "max_tokens": 7196,
                "stream": False
            }
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"直接 HTTP 請求也失敗: {e}")
            raise

    def classify_batch(self, messages: List[Dict[str, str]], output_path: str, batch_size: int = 35, max_context_tokens: int = 7196, max_retries: int = 1) -> None:
        total_batches = (len(messages) + batch_size - 1) // batch_size
        print(f"開始處理 {len(messages)} 則簡訊，分為 {total_batches} 個批次")
        failed_dir = Path(output_path).parent / "failed_batches"
        failed_dir.mkdir(exist_ok=True)
        header_written = False
        for batch_num in range(total_batches):
            print(f"處理第 {batch_num+1}/{total_batches} 批次", end="", flush=True)
            batch_messages = messages[batch_num *
                                      batch_size:(batch_num+1)*batch_size]
            success = False
            for retry in range(max_retries + 1):
                try:
                    start_time = time.time()
                    xml_input = self._create_messages_xml(batch_messages)
                    token_count = estimate_token_count(
                        self.prompt_content, batch_messages)
                    if token_count > max_context_tokens:
                        print(f" - 警告：context token 超過上限，跳過")
                        break
                    llm_response = self._call_llm(xml_input)
                    cleaned_response = sanitize_llm_xml(llm_response)
                    batch_results = parse_xml_response(
                        cleaned_response, tag="isTravel")
                    elapsed = time.time() - start_time
                    retry_info = f" (重試 {retry})" if retry > 0 else ""
                    print(
                        f" - 完成 ({len(batch_results)} 筆, {elapsed:.2f} 秒){retry_info}")
                    write_header = not header_written
                    with open(output_path, 'a', encoding='utf-8', newline='') as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(["sms_id", "label"])
                            header_written = True
                        for k, v in batch_results.items():
                            writer.writerow([k, v])
                    success = True
                    break
                except Exception as e:
                    if retry < max_retries:
                        print(f" - 錯誤 (將重試): {e}", end="", flush=True)
                        time.sleep(1)
                        continue
                    else:
                        print(f" - 錯誤 (已達最大重試次數): {e}")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        failed_file = failed_dir / \
                            f"failed_batch_{batch_num+1}_{timestamp}.txt"
                        try:
                            with open(failed_file, 'w', encoding='utf-8') as f:
                                f.write(f"批次編號: {batch_num+1}\n")
                                f.write(f"錯誤訊息: {e}\n")
                                f.write(f"輸入 XML:\n{xml_input}\n")
                                if 'llm_response' in locals():
                                    f.write(f"LLM 回應:\n{llm_response}\n")
                        except Exception as save_error:
                            print(f" - 無法儲存失敗批次資訊: {save_error}")
                        break
            if not success:
                print(f" - 批次 {batch_num+1} 處理失敗，已跳過")
                continue

    def process_file(self, input_path: str, output_dir: str = None) -> str:
        if output_dir is None:
            output_dir = Path(input_path).parent.parent / "results"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        existing_ids = get_existing_classified_ids(
            output_dir, "travel", self.model_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        safe_model_name = self.model_name.replace("/", "_").replace(":", "_")
        output_filename = f"travel_{safe_model_name}_{timestamp}.csv"
        output_path = Path(output_dir) / output_filename
        print(f"開始處理檔案: {input_path}")
        print(f"輸出路徑: {output_path}")
        all_messages = load_input_csv(input_path)
        print(f"載入 {len(all_messages)} 則簡訊")
        messages_to_process = [msg for msg in all_messages if str(
            msg["id"]) not in existing_ids]
        skipped_count = len(all_messages) - len(messages_to_process)
        if skipped_count > 0:
            print(f"跳過 {skipped_count} 則已分類簡訊")
        if not messages_to_process:
            print("所有簡訊已分類完成，無需處理")
            return str(output_path)
        print(f"剩餘待分類簡訊: {len(messages_to_process)} 則")
        self.classify_batch(messages_to_process, str(output_path))
        print(f"分類結果已儲存至: {output_path}")
        return str(output_path)


def main(model_name: str = "mistralai/magistral-small", base_url: str = "http://127.0.0.1:1234/v1", input_path: str = None, api_key: str = None):
    current_dir = Path(__file__).parent
    if input_path is None:
        input_path = current_dir / "data" / "input.csv"
    else:
        input_path = Path(input_path)
    if not input_path.exists():
        print(f"錯誤: 找不到輸入檔案 {input_path}")
        print("請確認檔案路徑是否正確")
        return
    try:
        classifier = TravelClassifier(
            model_name=model_name, base_url=base_url, api_key=api_key)
        output_path = classifier.process_file(str(input_path))
        print(f"\n✅ 分類任務完成！")
        print(f"📁 結果檔案: {output_path}")
    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    model_name = sys.argv[1] if len(
        sys.argv) > 1 else "mistralai/magistral-small"
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:1234/v1"
    input_path = sys.argv[3] if len(sys.argv) > 3 else None
    api_key = sys.argv[4] if len(sys.argv) > 4 else None
    main(model_name, base_url, input_path, api_key)
