# =============================
# PubMed 文献检索与解析脚本
# 使用 rentrez + xml2
# =============================

# 安装/加载依赖
if (!requireNamespace("rentrez", quietly = TRUE)) install.packages("rentrez")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("xml2", quietly = TRUE)) install.packages("xml2")
if (!requireNamespace("quanteda", quietly = TRUE)) install.packages("quanteda")
if (!requireNamespace("quanteda.textplots", quietly = TRUE)) install.packages("quanteda.textplots")

library(rentrez)
library(tidyverse)
library(xml2)
library(quanteda)
library(quanteda.textplots)

# 小工具：安全空值处理
`%||%` <- function(a, b) if (!is.null(a)) a else b

# 设定查询语句
query <- '(
  "Nature nanotechnology"[Journal] OR 
  "Journal of the American Chemical Society"[Journal] OR 
  "Angew Chem Int Ed Engl"[Journal] OR 
  "J Control Release"[Journal] OR 
  "Nat Biomed Eng"[Journal] OR 
  "Nat Commun"[Journal] OR 
  "Nanomedicine"[Journal] OR 
  "Adv Mater."[Journal] OR 
  "Nature materials"[Journal] OR 
  "ACS nano"[Journal] OR 
  "Nano letters"[Journal] OR 
  "Advanced Functional Materials"[Journal] OR 
  "Small"[Journal] OR 
  "Biomaterials"[Journal] OR 
  "Nano research"[Journal] OR 
  "ACS Applied Materials & Interfaces"[Journal] OR 
  "Nanoscale"[Journal] OR 
  "NPG Asia Materials"[Journal] OR 
  "Science advances"[Journal] OR 
  "Clinical Cancer Research"[Journal] OR 
  "Science"[Journal] OR 
  "Science Translational Medicine"[Journal] OR 
  "Cell Biomaterials"[Journal] OR 
  "Advanced Healthcare Materials"[Journal]
) AND (nanoparticles[All Fields]) AND (ROS[All Fields]) AND (ultrasound[All Fields]) AND (tumor[All Fields]) AND (cancer[All Fields])'

# API Key (可选，提高限额)
api_key <- "your_api_key_here"  # ←替换成你自己的
if (nchar(api_key) > 0) {
  options("ENTREZ_KEY" = api_key)
  options("rentrez.email" = "your_email@example.com") # ←换成你的邮箱
}

# ========== 函数部分 ==========
# 批量获取 PubMed IDs
fetch_pubmed_ids <- function(query, retmax = 10000) {
  search_res <- entrez_search(db = "pubmed", term = query, retmax = retmax, use_history = TRUE)
  return(search_res$ids)
}

# 根据 PMID 获取 XML 并解析
fetch_pubmed_records <- function(ids) {
  # 返回原始XML字符串
  xml_data <- entrez_fetch(db = "pubmed", id = ids, rettype = "xml", parsed = FALSE)
  doc <- read_xml(xml_data)
  records <- xml_find_all(doc, "//PubmedArticle")
  
  tibble(
    PMID     = xml_text(xml_find_first(records, ".//PMID")),
    Title    = xml_text(xml_find_first(records, ".//ArticleTitle")),
    Abstract = xml_text(xml_find_first(records, ".//Abstract/AbstractText")) %||% NA,
    Journal  = xml_text(xml_find_first(records, ".//Journal/Title")) %||% NA,
    PubDate  = xml_text(xml_find_first(records, ".//PubDate/Year")) %||% NA
  )
}


# ========== 主程序 ==========
cat("开始检索 PubMed...\n")
ids <- fetch_pubmed_ids(query, retmax = 1000)  # 建议分批获取，避免超限

if (length(ids) == 0) stop("没有检索到文献！")

cat("检索到", length(ids), "条记录，开始下载...\n")

# 批量解析（分组抓取，避免太大请求）

all_records <- map_dfr(split(ids, ceiling(seq_along(ids))), fetch_pubmed_records)

# 保存总结果
write.csv(all_records, "pubmed_results_2015_2025.csv", row.names = FALSE)
cat("总数据已保存到 pubmed_results_2015_2025.csv\n")

# 筛选 tumor-specific
tumor_specific_df <- all_records %>%
  filter(str_detect(Abstract, regex("glioma|breast|lung|pancreatic|colorectal|prostate|ovarian|liver|melanoma|leukemia|lymphoma|sarcoma|cancer|tumor|cervical", ignore_case = TRUE)) |
         str_detect(Title, regex("glioma|breast|lung|pancreatic|colorectal|prostate|ovarian|liver|melanoma|leukemia|lymphoma|sarcoma|cancer|tumor|cervical", ignore_case = TRUE)))


# ========== 可视化：词云 ==========
cat("生成摘要词云...\n")
corpus <- corpus(all_records$Abstract, docnames = all_records$PMID)
tokens <- tokens(corpus, remove_numbers = TRUE, remove_punct = TRUE, remove_symbols = TRUE)
dfm <- dfm(tokens) %>% dfm_trim(min_termfreq = 5)
textplot_wordcloud(dfm, max_words = 100, colors = RColorBrewer::brewer.pal(8, "Dark2"))

cat("处理完成！\n")
