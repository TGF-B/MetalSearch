import os
import pandas as pd
import requests
from tqdm import tqdm
import webbrowser
import time

CSV_FILE = "/home/donaldtangai4s/Desktop/ROS/new_query/pubmed_results_2015_2025.csv"
PDF_DIR = "/home/donaldtangai4s/Desktop/ROS/new_query/pdfs"
UNPAYWALL_EMAIL = "zhengqi-tang@shu.edu.cn"  # 请替换为你的邮箱

os.makedirs(PDF_DIR, exist_ok=True)
df = pd.read_csv(CSV_FILE)
if 'DOI' not in df.columns:
    df['DOI'] = ""
if 'status' not in df.columns:
    df['status'] = ""

def pmid_to_doi(pmid):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.ok:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(r.text)
            for article_id in root.findall(".//ArticleId"):
                if article_id.attrib.get("IdType") == "doi":
                    return article_id.text
    except Exception as e:
        print(f"PMID {pmid} 查DOI失败: {e}")
    return None

def title_to_doi(title):
    url = "https://api.crossref.org/works"
    params = {"query.title": title, "rows": 1}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.ok:
            items = r.json().get("message", {}).get("items", [])
            if items and "DOI" in items[0]:
                return items[0]["DOI"]
    except Exception as e:
        print(f"标题查DOI失败: {e}")
    return None

def get_oa_pdf_url(doi):
    url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    try:
        r = requests.get(url, timeout=15)
        if r.ok:
            data = r.json()
            loc = data.get("best_oa_location")
            if loc and loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
    except Exception as e:
        print(f"Unpaywall查询失败: {doi} -> {e}")
    return None

def download_pdf(pdf_url, save_path):
    try:
        r = requests.get(pdf_url, stream=True, timeout=30)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"{save_path} 下载失败: {e}")
        return False

# 1. 补全DOI
for idx, row in tqdm(df.iterrows(), total=len(df), desc="补全DOI"):
    doi = row.get('DOI')
    pmid = row.get('PMID')
    title = row.get('Title')
    if not doi or pd.isna(doi):
        if pmid and not pd.isna(pmid):
            doi = pmid_to_doi(str(pmid))
        if (not doi or pd.isna(doi)) and isinstance(title, str) and title.strip():
            doi = title_to_doi(title)
        if doi:
            df.at[idx, 'DOI'] = doi

# 2. 下载OA PDF
for idx, row in tqdm(df.iterrows(), total=len(df), desc="下载OA PDF"):
    doi = row.get('DOI')
    pmid = row.get('PMID')
    title = row.get('Title')
    if doi and not pd.isna(doi):
        pdf_url = get_oa_pdf_url(doi)
        pdf_filename = f"{pmid or idx}.pdf"
        save_path = os.path.join(PDF_DIR, pdf_filename)
        if pdf_url:
            success = download_pdf(pdf_url, save_path)
            df.at[idx, 'status'] = "downloaded" if success else "download_failed"
        else:
            df.at[idx, 'status'] = "no_oa_pdf"
    else:
        df.at[idx, 'status'] = "no_doi"

df.to_csv(CSV_FILE, index=False)
print("DOI补全和OA PDF下载完成，已保存")

# 3. 批量打开未下载的文献主页，人工下载
need_manual = df[df['status'] == "no_oa_pdf"]
batch_size = 10
for i in range(0, len(need_manual), batch_size):
    batch = need_manual.iloc[i:i+batch_size]
    print(f"\n请在浏览器中下载以下 {len(batch)} 篇文献（第{i+1}~{i+len(batch)}条）：")
    for _, row in batch.iterrows():
        doi = row['DOI']
        if doi and not pd.isna(doi):
            url = f"https://doi.org/{doi}"
            print(url)
            webbrowser.open(url)
        else:
            print(f"无DOI，无法打开：{row.get('Title','')}")
    input("请下载完本批次后按回车继续...")
    # 可选：等待人工下载后，手动修改status为downloaded

print("全部处理完成。")