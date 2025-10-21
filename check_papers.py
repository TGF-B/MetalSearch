import os
import re
import pandas as pd

CSV_FILE = "/home/donaldtangai4s/Desktop/ROS/new_query/pubmed_results_2015_2025.csv"
FULLTEXT_DIR = "/home/donaldtangai4s/Desktop/ROS/new_query/FullText"

def normalize_to_words(text):
    """清洗文本并拆分为词集合"""
    text = text.lower()
    text = re.sub(r'[\s\W_]+', ' ', text)  # 非字母数字替换为空格
    words = set(text.split())
    return words

def update_status():
    # 读取 CSV
    df = pd.read_csv(CSV_FILE)
    if "status" not in df.columns:
        df["status"] = ""

    # 获取 PDF 文件名（无后缀）
    pdf_files = [f for f in os.listdir(FULLTEXT_DIR) if f.lower().endswith(".pdf")]
    pdf_basenames = [os.path.splitext(f)[0] for f in pdf_files]
    pdf_wordsets = {name: normalize_to_words(name) for name in pdf_basenames}

    # 遍历 CSV 中的标题
    for idx, row in df.iterrows():
        title = str(row["Title"])
        title_words = normalize_to_words(title)

        found = False
        for pdf_name, pdf_words in pdf_wordsets.items():
            if not title_words:
                continue
            overlap = len(title_words & pdf_words) / len(title_words)
            if overlap >= 0.5:  # 至少一半词匹配上
                found = True
                break

        df.at[idx, "status"] = "downloaded" if found else "not_downloaded"

    # 保存更新后的 CSV
    df.to_csv(CSV_FILE, index=False)

    # ===== 统计报告 =====
    total = len(df)
    downloaded = (df["status"] == "downloaded").sum()
    not_downloaded = (df["status"].astype(str).str.lower() == "not_downloaded").sum()

    print("===== Status Report =====")
    print(f"Total articles     : {total}")
    print(f"Downloaded         : {downloaded} ({downloaded/total:.2%})")
    print(f"Not downloaded     : {not_downloaded} ({not_downloaded/total:.2%})")
    print("Updated CSV saved:", CSV_FILE)

if __name__ == "__main__":
    update_status()
