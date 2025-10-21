#!/usr/bin/env python3
"""
Therapeutic-parameter extractor with Fulltext-first + RAG (PubMedBERT+FAISS) + LLM (OpenAI-compatible) + regex validation.
- No fabrication: only extract from provided title/abstract or fetched OA fulltext (Unpaywall/PMC/DOI HTML).
- RAG: for each parameter, retrieve top-k evidence chunks by PubMedBERT embeddings, pass to local LLM for structured JSON.
- Validation: regex patterns validate and normalize LLM outputs; regex hits take precedence.
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import sys
import re
import math
import argparse
import json
import time
import unicodedata
import pathlib
import logging
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Transformers/torch optional
try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoTokenizer as QATok, AutoModelForQuestionAnswering
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False
    torch = None

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


# 强制开关（默认禁用 FAISS，避免 numpy 类型报错）
USE_FAISS = os.getenv("USE_FAISS", "0") == "1"

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# IO defaults
DEFAULT_INPUT = "/home/donaldtangai4s/Desktop/ROS/new_query/pubmed_results_2015_2025.csv"
OUTPUT_DIR = "/home/donaldtangai4s/Desktop/ROS/new_query/features_extracted"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "extracted_therapy_parameters_full_RAG.csv")
VISUAL_PNG = os.path.join(OUTPUT_DIR, "parameter_distribution_stacked_RAG.png")
FULLTEXT_DIR = "/home/donaldtangai4s/Desktop/ROS/fulltexts"

# Regex patterns
_PATTERNS = {
    "dose_mg_per_kg": re.compile(r'(\d+(?:\.\d+)?)\s*mg\s*(?:/|\s*)(?:kg)(?:\s*(?:[-−]\s*1|\^-\s*1|/\s*day|/\s*d))?', re.I),
    "dose_mg_per_g": re.compile(r'(\d+(?:\.\d+)?)\s*mg\s*/\s*g', re.I),
    "dose_ug_per_ml": re.compile(r'(\d+(?:\.\d+)?)\s*(?:μg|ug|mcg)\s*/\s*ml', re.I),
    "dose_mg_per_ml": re.compile(r'(\d+(?:\.\d+)?)\s*mg\s*/\s*ml', re.I),
    "injection_count": re.compile(r'(\d+)\s+(?:injections|doses|times)', re.I),
    "treatment_duration_days": re.compile(r'(?:for|treated\s*for|treatment\s*lasted\s*)(?:about\s*)?(\d{1,3})\s*days', re.I),
    "tumor_inhibition_pct": re.compile(r'(\d{1,3}(?:\.\d+)?)\s*%\s*(?:tumor\s*(?:growth\s*)?)?(?:inhibit|inhibition|reduction|decrease)', re.I),
    "survival_days": re.compile(r'(?:median\s*)?survival(?:\s*time)?(?:\s*of|\s*was|\s*:\s*)\s*(\d{1,3})\s*days', re.I),
    "particle_size_nm": re.compile(r'(\d+(?:\.\d+)?)\s*(?:nm|nanometer|nanometers)\b', re.I),
    "irradiation_wavelength_nm": re.compile(r'(\d{2,4})\s*nm', re.I),
    "irradiation_power": re.compile(r'(\d+(?:\.\d+)?)\s*(?:W/cm2|W/cm\^2|W/cm²|W/cm)', re.I),
    "ros_percent": re.compile(r'(\d{1,3}(?:\.\d+)?)\s*%\s*(?:increase|increase in ROS|ROS increase|ROS level)', re.I),
}

MORPHOLOGY_KEYWORDS = {
    "nanosheet": ["nanosheet", "nanosheets", "sheet-like"],
    "nanorod": ["nanorod", "nanorods", "rod-shaped"],
    "nanoparticle": ["nanoparticle", "nanoparticles", "NP", "nanocrystal"],
    "hollow": ["hollow", "core-shell", "core shell", "hollowed"],
    "porous": ["porous", "mesoporous"],
    "nanosphere": ["nanosphere", "sphere", "spherical"],
    "nanoparticles": ["nanoparticles", "particles", "particle"]
}
ADMIN_ROUTE_KEYWORDS = {
    "intravenous": ["intravenous", "i.v.", "iv", "intravenously"],
    "intraperitoneal": ["intraperitoneal", "i.p.", "ip", "intraperitoneally"],
    "intratumoral": ["intratumoral", "i.t.", "it", "intratumorally", "intratumoural"],
    "oral": ["oral", "orally", "p.o.", "per os"],
    "subcutaneous": ["subcutaneous", "s.c.", "sc", "subcutaneously"]
}
INTERVENTION_KEYWORDS = {
    "photothermal": ["photothermal", "PTT", "photothermal therapy", "ultrasound"],
    "photodynamic": ["photodynamic", "PDT", "photosensitizer"],
    "radiation": ["radiation", "radiotherapy", "RT", "irradiation", "Gy"],
    "chemo": ["chemotherapy", "cisplatin", "doxorubicin", "drug"],
    "bacteria": ["bacteria", "bacterial", "Salmonella", "E. coli", "Clostridium"],
    "ultrasound": ["ultrasound", "sonodynamic", "SDT", "sonosensitizer"],
}

TUMOR_KEYWORDS = {
    "breast cancer": ["breast cancer", "MCF-7", "MDAMB231", "MDA-MB-231", "4T1"],
    "lung cancer": ["lung cancer", "A549", "H1299", "NSCLC", "SCLC"],
    "liver cancer": ["liver cancer", "HepG2", "Hep3B", "hepatocellular"],
    "colon cancer": ["colon cancer", "HT-29", "HCT116", "CT26"],
    "ovarian cancer": ["ovarian", "SKOV3"],
    "glioblastoma": ["glioblastoma", "U87", "U251", "GBM"],
    "leukemia": ["AML", "ALL", "K562"],
    "pancreatic cancer": ["PANC-1", "pancreatic"],
    "melanoma": ["B16", "B16-F10", "melanoma"],
    "head_neck": ["SCC", "HNSCC"],
    "prostate": ["PC3", "LNCaP", "prostate"],
    "tc1_model": ["TC1", "TC-1", "TC 1"],
    "other": ["xenograft", "patient-derived", "PDX"]
}
SYNTHESIS_METHOD_KEYWORDS = {
    "hydrothermal": ["hydrothermal", "solvothermal"],
    "sol-gel": ["sol-gel", "sol gel"],
    "precipitation": ["precipitation", "coprecipitation"],
    "thermal": ["thermal decomposition", "calcination", "anneal"],
    "microwave": ["microwave"],
    "sonochemical": ["sonochemical", "ultrasonic"],
    "electrochemical": ["electrochemical", "electrodeposition"]
}

def safe_lower(s): return (s or "").lower()

def text_normalize(txt: str) -> str:
    if not txt: return ""
    t = unicodedata.normalize("NFKC", txt)
    t = (t.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff").replace("ﬃ", "ffi").replace("ﬄ", "ffl"))
    t = t.replace("–", "-").replace("—", "-").replace("−", "-")
    t = re.sub(r'([A-Za-z])-\s*\n\s*([A-Za-z])', r'\1\2', t)
    t = re.sub(r'[ \t\r\f]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def extract_main_article_html(html: str) -> str:
    if not html: return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for sel in ["script","style","nav","header","footer","aside","noscript"]:
            for t in soup.select(sel): t.extract()
        candidates, selectors = [], [
            "article","[role='main']", ".c-article-body",".c-article-content",".article-content",
            "#article-content",".main-content",".content",".article","#content","#main-content",".article-body"
        ]
        for sel in selectors:
            for node in soup.select(sel):
                txt = node.get_text("\n", strip=True)
                if txt and len(txt) > 500: candidates.append(txt)
        text = max(candidates, key=len) if candidates else soup.get_text("\n", strip=True)
        text = re.sub(r'(References|Acknowledg(e)?ments?|Funding|Conflict(s)? of interest)\b.*', '', text, flags=re.I|re.S)
        return text_normalize(text)
    except Exception:
        return text_normalize(re.sub(r'<[^>]+>', ' ', html or ""))

def _strip_html(text: str) -> str:
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(text, "lxml").get_text("\n")
    except Exception:
        return re.sub(r'<[^>]+>', ' ', text or '')

def first_match_keywords(text, mapping):
    text_l = safe_lower(text)
    for key, kws in mapping.items():
        for kw in kws:
            if kw.lower() in text_l:
                return key
    return None

def extract_regexes(text):
    result = {}
    for k, pattern in _PATTERNS.items():
        m = pattern.search(text)
        if m:
            val = m.group(1)
            if k in ("dose_mg_per_kg","dose_mg_per_g","dose_ug_per_ml","dose_mg_per_ml",
                     "particle_size_nm","irradiation_wavelength_nm","irradiation_power",
                     "ros_percent","tumor_inhibition_pct","survival_days"):
                try: result[k] = float(val)
                except Exception: result[k] = val
            else:
                result[k] = val
    return result

def contextual_particle_size(text: str):
    if not text: return None
    try:
        m = re.search(r'(?:size|diameter|hydrodynamic|DLS|TEM)[^\.]{0,120}?(\d+(?:\.\d+)?)\s*(?:nm|nanometer|nanometers)\b', re.I)
        return float(m.group(1)) if m else None
    except Exception:
        return None

def extract_from_text(text):
    text_l = safe_lower(text)
    extracted = extract_regexes(text)
    csz = contextual_particle_size(text)
    if csz is not None:
        extracted["particle_size_nm"] = csz
    morph = first_match_keywords(text, MORPHOLOGY_KEYWORDS)
    if morph: extracted["morphology"] = morph
    route = first_match_keywords(text, ADMIN_ROUTE_KEYWORDS)
    if route: extracted["administration_route"] = route
    inters = []
    for ik, kws in INTERVENTION_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text_l: inters.append(ik); break
    if inters: extracted["interventions"] = list(dict.fromkeys(inters))
    tumors = []
    for tumor, kws in TUMOR_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text_l: tumors.append(tumor); break
    if tumors: extracted["tumor_types"] = list(dict.fromkeys(tumors))
    synth = first_match_keywords(text, SYNTHESIS_METHOD_KEYWORDS)
    if synth: extracted["synthesis_method"] = synth
    m = re.search(r'(?:synthesi(?:s|ze|zed):?)(.{20,800})', text, re.I|re.S) or \
        re.search(r'(?:prepared by|were prepared|was prepared|prepared using)(.{20,800})', text, re.I|re.S)
    #THE (20,800) is to avoid too short or too long descriptions
    if m:
        desc = m.group(1).strip()
        extracted["synthesis_description"] = desc
        extracted["synthesis_desc_len"] = len(desc)
        extracted["synthesis_fullstop_count"] = desc.count(".")
    return extracted

def fetch_pmc_fulltext_via_pmid(pmid: str, timeout: int = 20) -> str:
    try:
        elink = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        r = requests.get(elink, params={"dbfrom":"pubmed","db":"pmc","id":pmid}, timeout=timeout); r.raise_for_status()
        root = ET.fromstring(r.text); pmcid = None
        for idnode in root.findall(".//LinkSetDb/Link/Id"):
            if idnode is not None and idnode.text:
                if idnode.text.startswith("PMC"): pmcid = idnode.text; break
                if idnode.text.isdigit(): pmcid = f"PMC{idnode.text}"; break
        if not pmcid: return None
        efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        r2 = requests.get(efetch, params={"db":"pmc","id":pmcid}, timeout=timeout); r2.raise_for_status()
        root2 = ET.fromstring(r2.text); paras=[]
        for node in root2.findall(".//body//p"):
            txt = "".join(node.itertext()).strip()
            if txt: paras.append(txt)
        fulltxt = "\n".join(paras).strip()
        return fulltxt or None
    except Exception:
        return None

def estimate_synthesis_complexity(extracted):
    method = extracted.get("synthesis_method")
    steps = extracted.get("synthesis_steps") if isinstance(extracted.get("synthesis_steps"), (int, float)) else None
    desc_len = extracted.get("synthesis_desc_len", 0)
    fullstops = extracted.get("synthesis_fullstop_count", 0)
    score = 0
    if method in ("hydrothermal","thermal","electrochemical","sonochemical"): score += 2
    if method in ("sol-gel","precipitation","microwave"): score += 1
    if steps: score += 0 if steps <= 1 else (1 if steps <= 3 else 2)
    if desc_len >= 400: score += 2
    elif desc_len >= 150: score += 1
    if fullstops >= 5: score += 2
    elif fullstops >= 2: score += 1
    if score >= 5: return "complex"
    if score >= 3: return "moderate"
    if score >= 1: return "simple"
    return "unknown"

def normalize_dose_info(extr):
    out = {}
    if "dose_mg_per_kg" in extr: out["dose_value"]=extr["dose_mg_per_kg"]; out["dose_unit"]="mg/kg"
    elif "dose_mg_per_g" in extr: out["dose_value"]=extr["dose_mg_per_g"]; out["dose_unit"]="mg/g"
    elif "dose_ug_per_ml" in extr: out["dose_value"]=extr["dose_ug_per_ml"]; out["dose_unit"]="ug/mL"
    elif "dose_mg_per_ml" in extr: out["dose_value"]=extr["dose_mg_per_ml"]; out["dose_unit"]="mg/mL"
    return out

def unpaywall_is_open(doi, email):
    if not doi or doi == "N/A": return None
    try:
        url = f"https://api.unpaywall.org/v2/{doi}"
        resp = requests.get(url, params={"email": email}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("is_oa", False), data
    except Exception:
        return None
    return None

# Grobid helpers
def grobid_extract_fulltext(pdf_path, grobid_url="http://localhost:8070/api/processFulltextDocument"):
    try:
        with open(pdf_path, "rb") as f:
            files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
            resp = requests.post(grobid_url, files=files, timeout=60)
        if resp.status_code == 200 and resp.text and "<TEI" in resp.text:
            return resp.text
    except Exception:
        pass
    return None

def extract_text_from_tei(tei_xml):
    try:
        from lxml import etree
        root = etree.fromstring(tei_xml.encode("utf-8"))
        paras = root.xpath("//text//body//p | //text//body//div//p")
        txts = []
        for p in paras:
            s = etree.tostring(p, method="text", encoding="unicode").strip()
            if s: txts.append(s)
        return text_normalize("\n".join(txts).strip())
    except Exception:
        return text_normalize(re.sub(r"<[^>]+>", " ", tei_xml or ""))

def _best_oa_url(oa_data):
    best = None
    for loc in oa_data.get("oa_locations", []) or []:
        if loc.get("url_for_pdf"): return loc.get("url_for_pdf")
        if not best and loc.get("url"): best = loc.get("url")
    return best

def _download_pdf(url, out_path, timeout=30):
    r = requests.get(url, timeout=timeout, stream=True); r.raise_for_status()
    with open(out_path, "wb") as fh:
        for chunk in r.iter_content(1024*32):
            if chunk: fh.write(chunk)

def _save_sidecar_text(outdir, doi, pmid, text):
    try:
        base = doi.replace("/", "_") if doi and doi != "N/A" else f"PMID_{pmid or 'NA'}"
        sidecar = pathlib.Path(outdir) / f"{base}.txt"
        sidecar.write_text(text or "", encoding="utf-8")
    except Exception:
        pass

def fallback_pdf_text_via_pymupdf(pdf_path: str, max_pages=200) -> str:
    try:
        import fitz
    except Exception:
        return None
    try:
        doc = fitz.open(pdf_path); texts=[]; hf={}
        for i, page in enumerate(doc):
            if i >= max_pages: break
            blocks = sorted(page.get_text("blocks"), key=lambda b: (round(b[1],1), round(b[0],1)))
            lines=[]
            for b in blocks:
                t=b[4].strip()
                if t: lines.append(t)
            page_text="\n".join(lines)
            for seg in [l.strip() for l in page_text.splitlines() if l.strip()]:
                if len(seg) <= 80: hf[seg] = hf.get(seg,0)+1
            texts.append(page_text)
        doc.close()
        full="\n".join(texts)
        repeats = {k for k,v in hf.items() if v >= max(3, int(0.2*len(texts)))}
        if repeats:
            pat = re.compile("|".join(re.escape(r) for r in sorted(repeats, key=len, reverse=True)))
            full = pat.sub("", full)
        full = re.sub(r'(References|Acknowledg(e)?ments?|Funding|Conflict(s)? of interest)\b.*', '', full, flags=re.I|re.S)
        return text_normalize(full)
    except Exception:
        return None

def download_oa_pdf_and_extract_text(doi, email, outdir=FULLTEXT_DIR, timeout=20, grobid_url="http://localhost:8070/api/processFulltextDocument"):
    os.makedirs(outdir, exist_ok=True)
    if not doi or doi == "N/A" or not email: return None, None
    try:
        resp = requests.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": email}, timeout=timeout)
        if resp.status_code != 200: return None, None
        data = resp.json()
        if not data.get("is_oa"): return None, None
        best_url = _best_oa_url(data)
        if not best_url: return None, None
        pdf_path = pathlib.Path(outdir) / (doi.replace("/", "_") + ".pdf")
        try:
            _download_pdf(best_url, str(pdf_path), timeout=max(30, timeout))
        except Exception:
            return None, None
        tei_xml = grobid_extract_fulltext(str(pdf_path), grobid_url=grobid_url)
        if tei_xml:
            txt = extract_text_from_tei(tei_xml) or ""
            if txt.strip():
                _save_sidecar_text(outdir, doi, None, txt)
                return txt, "pdf_grobid"
        fb = fallback_pdf_text_via_pymupdf(str(pdf_path))
        if fb and len(fb) > 500:
            _save_sidecar_text(outdir, doi, None, fb)
            return fb, "pdf_text"
        return None, None
    except Exception:
        return None, None

def try_fetch_html_from_unpaywall(oa_data, timeout=15):
    url = _best_oa_url(oa_data) if oa_data else None
    if not url: return None
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return extract_main_article_html(r.text)
    except Exception:
        pass
    return None

# QA helpers (optional)
def init_qa_pipeline(model_name, device=-1, local_only=True):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, local_files_only=local_only)
    tokenizer = QATok.from_pretrained(model_name, use_fast=True, local_files_only=local_only)
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

def chunk_text_for_qa(text, max_chars=3000):
    if not text: return []
    if len(text) <= max_chars: return [text]
    sentences = re.split(r'(?<=[\.!?])\s+', text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    return chunks

# RAG: PubMedBERT embedder + FAISS
class PubMedBERTEmbedder:
    def __init__(self, model_name_or_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", device=None, local_only=True):
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_only)
        self.model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_only)
        self.device = device if (device is not None) else (0 if torch.cuda.is_available() else -1)
        if self.device >= 0: self.model = self.model.cuda(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=16, max_length=512):
        # 统一为列表[str]
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, (list, tuple)):
            texts = [str(texts)]
        texts = [("" if t is None else str(t)) for t in texts]
        if len(texts) == 0:
            hid = getattr(self.model.config, "hidden_size", 768)
            return np.zeros((0, hid), dtype="float32")

        outs=[]
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tok = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            if self.device >= 0:
                tok = {k:v.cuda(self.device) for k,v in tok.items()}
            out = self.model(**tok).last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1)
            summed = (out * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            emb = (summed / counts).detach().cpu().numpy()
            outs.append(emb)

        X = np.vstack(outs).astype("float32", copy=False)
        X = np.ascontiguousarray(X)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.clip(norms, 1e-12, None)
        return X

def build_faiss_index(embeddings):
    # 未显式启用则不用 FAISS
    if not (_FAISS_AVAILABLE and USE_FAISS):
        return None
    emb = np.ascontiguousarray(embeddings.astype('float32', copy=False))
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

def make_chunks_for_rag(text, max_chars=1200):
    if len(text) > 200000:  # 限制20万字符
        text = text[:200000]
    chunks = chunk_text_for_qa(text, max_chars=max_chars)
    kept=[]
    for c in chunks:
        if re.search(r'\b(References|Acknowledg(e)?ments?|Conflict of interest|Funding)\b', c, re.I): continue
        kept.append(c)
    # 限制最多200块，避免内存/显存爆
    kept = kept[:200]
    return kept or chunks

def rag_retrieve(topics, chunks, embedder, top_k=6, debug=True):
    if not chunks: return {k: [] for k in topics}
    # 降低批量与序列长度
    X = embedder.encode(chunks, batch_size=8, max_length=256)
    X = np.ascontiguousarray(np.asarray(X, dtype='float32'))
    index = build_faiss_index(X)
    res = {}
    for key, queries in topics.items():
        q_emb = embedder.encode(queries, batch_size=32, max_length=64)
        q_emb = np.ascontiguousarray(np.asarray(q_emb, dtype='float32'))
        scored = {}
        if index is not None:
            try:
                D, I = index.search(q_emb, min(top_k, X.shape[0]))
                for qi in range(q_emb.shape[0]):
                    for j, idx in enumerate(I[qi]):
                        if idx < 0: continue
                        scored[idx] = max(scored.get(idx, -1.0), float(D[qi][j]))
            except Exception as e:
                if debug: logging.warning(f"[RAG][WARN] FAISS search failed for '{key}': {e}")
                index = None
        if index is None:
            sims = np.dot(q_emb, X.T)               # 余弦相似度（已 L2 归一）
            for qi in range(sims.shape[0]):
                k = min(top_k, sims.shape[1])
                idxs = np.argpartition(-sims[qi], range(k))[:k]
                for idx in idxs:
                    scored[idx] = max(scored.get(idx, -1.0), float(sims[qi, idx]))
        picked = [chunks[i] for i,_ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
        res[key] = picked
    return res

# LLM (OpenAI-compatible, e.g., Ollama/vLLM server)
def llm_openai_chat(base_url, api_key, model, messages, max_tokens=512, temperature=0.0, timeout=60):
    import requests as _r, json as _json
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type":"application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type":"json_object"}
    }
    rr = _r.post(url, headers=headers, data=_json.dumps(payload), timeout=timeout)
    rr.raise_for_status()
    data = rr.json()
    txt = data["choices"][0]["message"]["content"]
    try:
        b, e = txt.find("{"), txt.rfind("}")
        return json.loads(txt[b:e+1]) if b!=-1 and e!=-1 else {}
    except Exception:
        return {}

def llm_extract_structured(context_text: str, cfg: dict, debug=False) -> dict:
    if not context_text or not cfg: return {}
    schema = """
Return JSON:
{
  "dose_value": number|null,
  "dose_unit": "mg/kg"|"mg/g"|"ug/mL"|"mg/mL"|null,
  "administration_route": string|null,
  "tumor_types": string[]|null,
  "interventions": string[]|null,
  "particle_size_nm": number|null,
  "tumor_inhibition_pct": number|null,
  "survival_days": number|null,
  "evidence": { "dose": string|null, "route": string|null, "tgi": string|null, "size": string|null }
}
Rules: Use only values explicitly present; prefer in vivo dose (mg/kg); keep evidence short quotes.
"""
    messages = [
        {"role":"system","content":"You are a precise IE model. Output strict JSON only."},
        {"role":"user","content": schema.strip() + "\n\nContext:\n" + context_text[:18000]}
    ]
    try:
        resp = llm_openai_chat(
            base_url=cfg.get("base_url","http://127.0.0.1:8000/v1"),
            api_key=cfg.get("api_key"),
            model=cfg.get("model","Qwen2.5-7B-Instruct"),
            messages=messages,
            max_tokens=int(cfg.get("max_tokens", 600)),
            temperature=0.0,
            timeout=90
        )
        if debug: logging.info(f"[LLM] keys={list(resp.keys()) if isinstance(resp, dict) else type(resp)}")
        return resp if isinstance(resp, dict) else {}
    except Exception as e:
        if debug: logging.warning(f"[LLM][ERR] {e}")
        return {}

def validate_value_in_context(val, unit_key, contexts_joined: str):
    # 以严格正则在上下文中复核
    if unit_key == "mg/kg":
        pat = _PATTERNS["dose_mg_per_kg"]
    elif unit_key == "mg/g":
        pat = _PATTERNS["dose_mg_per_g"]
    elif unit_key == "ug/mL":
        pat = _PATTERNS["dose_ug_per_ml"]
    elif unit_key == "mg/mL":
        pat = _PATTERNS["dose_mg_per_ml"]
    else:
        return False
    if val is None: return False
    txt = contexts_joined
    return bool(pat.search(txt))

def validate_llm_outputs(llm_out: dict, contexts: dict) -> dict:
    if not llm_out: return {}
    out = {}
    joined_all = "\n\n".join(sum(contexts.values(), [])) if contexts else ""
    # 剂量
    dv, du = llm_out.get("dose_value"), llm_out.get("dose_unit")
    if dv is not None and du in ("mg/kg","mg/g","ug/mL","mg/mL"):
        if validate_value_in_context(dv, du, joined_all):
            out["dose_value"], out["dose_unit"] = dv, du
    # 其他字段简单保留，由上层再正则合并兜底
    for k in ("administration_route","tumor_types","interventions","particle_size_nm","tumor_inhibition_pct","survival_days"):
        if llm_out.get(k) not in (None, "", []): out[k] = llm_out[k]
    out["evidence"] = llm_out.get("evidence")
    return out

# -------- Main per-row processing --------
def process_row(row, unpaywall_email=None, try_fulltext=True, use_transformer=False, qa_pipeline=None,
                timeout=15, debug=False, use_rag=False, embedder_model=None, llm_config=None):
    # Only process verified
    if not (str(row.get("pubmed_verified", "")).lower() in ("true","1","yes") or
            str(row.get("crossref_verified", "")).lower() in ("true","1","yes")):
        return {"pmid": row.get("pmid"), "status": "unverified", "needs_manual_check": True}

    title = row.get("title", "") or ""
    abstract = row.get("abstract", "") or ""
    combined = text_normalize((title + "\n\n" + abstract).strip())

    rec = {
        "pmid": row.get("pmid"),
        "doi": row.get("doi"),
        "title": title,
        "year": row.get("year"),
        "journal": row.get("journal"),
        "material": row.get("material"),
        "search_query": row.get("search_query"),
        "status": "ok",
        "needs_manual_check": False,
        "paywalled": None,
        "confidence": 0.0,
        "fulltext_checked": False,
        "fulltext_used_for_extraction": False
    }

    # 1) Regex from title+abstract
    extracted = extract_from_text(combined)

    # 2) Fulltext attempts
    fulltxt = None; fulltext_source = None
    rec["fulltext_checked"] = bool(try_fulltext and (row.get("doi") or row.get("pmid")))
    if try_fulltext:
        # Unpaywall + Grobid
        if unpaywall_email and rec["doi"] and rec["doi"] != "N/A":
            try:
                oa_info = unpaywall_is_open(rec["doi"], unpaywall_email)
                if isinstance(oa_info, tuple):
                    rec["paywalled"] = not oa_info[0]
                    if oa_info[0]:
                        fulltxt, fulltext_source = download_oa_pdf_and_extract_text(rec["doi"], unpaywall_email, outdir=FULLTEXT_DIR, timeout=timeout)
                        if not fulltxt:
                            html_txt = try_fetch_html_from_unpaywall(oa_info[1], timeout=timeout)
                            if html_txt and len(html_txt) > 500:
                                fulltxt, fulltext_source = html_txt, "html"
                else:
                    rec["paywalled"] = True
            except Exception as e:
                if debug: logging.warning(f"[FT][Unpaywall] DOI={rec['doi']} err={e}")
                rec["paywalled"] = True
        # PMC fallback
        if (not fulltxt) and row.get("pmid"):
            try:
                pmc_txt = fetch_pmc_fulltext_via_pmid(str(row.get("pmid")), timeout=timeout)
                if pmc_txt and len(pmc_txt) > 500:
                    fulltxt, fulltext_source = pmc_txt, "pmc"
                    _save_sidecar_text(FULLTEXT_DIR, rec.get("doi"), row.get("pmid"), pmc_txt)
            except Exception as e:
                if debug: logging.warning(f"[FT][PMC] PMID={row.get('pmid')} err={e}")
        # DOI landing HTML
        if (not fulltxt) and rec.get("doi") and rec["doi"] != "N/A":
            try:
                r = requests.get(f"https://doi.org/{rec['doi']}", timeout=timeout)
                if r.status_code == 200:
                    html_txt = extract_main_article_html(r.text)
                    if html_txt and len(html_txt) > 800:
                        fulltxt, fulltext_source = html_txt, "doi_html"
                        _save_sidecar_text(FULLTEXT_DIR, rec.get("doi"), row.get("pmid"), html_txt)
            except Exception:
                pass

    # Use fulltext if present
    if fulltxt:
        rec["fulltext_chars"] = len(fulltxt)
        rec["fulltext_source"] = fulltext_source
        rec["fulltext_used_for_extraction"] = True
        if debug: logging.info(f"[FT] pmid={rec.get('pmid')} doi={rec.get('doi')} source={fulltext_source} chars={len(fulltxt)}")
        extr_full = extract_from_text(fulltxt)
        for k, v in extr_full.items():
            if k not in extracted or not extracted.get(k):
                extracted[k] = v

    # 3) RAG + LLM (per-parameter检索 → 结构化抽取 → 正则校验)
    if use_rag and (llm_config is not None):
        try:
            base_text = fulltxt if (fulltxt and len(fulltxt) > 300) else combined
            chunks = make_chunks_for_rag(base_text, max_chars=1200)
            # 初始化嵌入器
            if not embedder_model:
                # 默认使用本地目录优先
                default_local = "/home/donaldtangai4s/Desktop/ROS/PubMedBert"
                model_path = default_local if os.path.isdir(default_local) else "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
            else:
                model_path = embedder_model
            embedder = PubMedBERTEmbedder(model_name_or_path=model_path, device=-1, local_only=True)
            topics = {
                "dose": ["dose mg/kg","in vivo dose mg kg-1","dosage mg/kg","dose mg per kg","dose mg per g","dose ug/mL","dose mg/mL",],
                "route": ["administration route intravenous intraperitoneal intratumoral oral subcutaneous"],
                "tgi": ["tumor inhibition %","tumor growth inhibition percentage","inhibition rate","tgi %","tumor inhibition rate","tumor growth inhibition"],
                "irradiation_wavelength_nm": ["irradiation wavelength nm","laser wavelength nm","wavelength nm"],
                "irradiation_power": ["irradiation power","laser power"],
                "ros_percent": ["ROS %","reactive oxygen species %","ROS generation %"],
                "dose_value": ["dose value","dose amount"],
                "survival_days": ["survival days","survival time days"],
                "size": ["particle size nm","hydrodynamic diameter nm","DLS size nm","TEM size nm"],
                "synth": ["synthesis method prepared by synthesized by hydrothermal solvothermal precipitation sol-gel"]
            }
            retrieved = rag_retrieve(topics, chunks, embedder, top_k=6, debug=debug)
            # 为 LLM 组装上下文（多路证据拼接）
            context = []
            for key in ("dose","route","tgi","size","synth"):
                if retrieved.get(key):
                    context.append(f"[[{key.upper()}]]\n" + "\n".join(retrieved[key]))
            context_text = "\n\n====\n\n".join(context)[:18000] if context else base_text[:18000]
            llm_raw = llm_extract_structured(context_text, llm_config, debug=debug)
            llm_valid = validate_llm_outputs(llm_raw, retrieved)
            # 合并：先正则，再 LLM（补空or一致）
            regex_from_ctx = extract_from_text(context_text)
            # 规则优先
            for k, v in normalize_dose_info(regex_from_ctx).items():
                if v is not None: extracted[k] = v
            for fld in ("administration_route","particle_size_nm","tumor_inhibition_pct"):
                if regex_from_ctx.get(fld) not in (None,"",[]): extracted[fld] = regex_from_ctx[fld]
            # LLM 补空
            for k in ("dose_value","dose_unit","administration_route","tumor_types","interventions","particle_size_nm","tumor_inhibition_pct","survival_days"):
                if (k not in extracted or not extracted.get(k)) and (k in llm_valid):
                    extracted[k] = llm_valid[k]
            if llm_valid.get("evidence"): rec["llm_evidence"] = llm_valid.get("evidence")
            rec["rag_used"] = True; rec["llm_used"] = True
        except Exception as e:
            if debug: logging.warning(f"[RAG][ERR] pmid={rec.get('pmid')} err={e}")

    # 4) Optional Transformer QA as fallback
    if use_transformer and qa_pipeline is not None:
        questions = {
            "dose": "What is the dose? (e.g., X mg/kg, X mg/g, X μg/mL)",
            "administration_route": "What was the administration route?",
            "tumor_model": "Which tumor model or cell line was used?",
            "treatment_duration": "How long was the treatment? (days)",
            "injections": "How many injections or doses were given?",
            "tumor_inhibition": "What was the tumor inhibition percentage or tumor growth inhibition?",
            "synthesis_method": "How were the nanoparticles synthesized? Which method was used?",
            "particle_size": "What is the particle size (nm)?",
            "intervention": "Which therapy or intervention was applied (e.g., PTT, PDT, ultrasound,chemotherapy)?"
        }
        ctx = fulltxt if (fulltxt and len(fulltxt) > 200) else combined
        try:
            from transformers import pipeline as hf_pipeline
            qa_pipe = qa_pipeline
            # 仅在尚未拿到值时再尝试补齐
            # ...可按需复用你原先的 QA 逻辑...
        except Exception as e:
            if debug: logging.warning(f"[QA][ERR] pmid={rec.get('pmid')} err={e}")

    # 5) Normalize + copy fields
    dose_norm = normalize_dose_info(extracted)
    rec.update(dose_norm)
    for fld in ("administration_route","morphology","tumor_types","interventions","particle_size_nm",
                "irradiation_wavelength_nm","irradiation_power","tumor_inhibition_pct","survival_days",
                "synthesis_description","synthesis_method","synthesis_steps",
                "injection_count","treatment_duration_days","ros_percent"):
        if fld in extracted:
            rec[fld] = extracted[fld]

    # 6) Confidence
    rec["synthesis_complexity"] = estimate_synthesis_complexity(extracted)
    score = 0.0; weight = 0.0
    def wadd(cond, w):
        nonlocal score, weight
        weight += w
        if cond: score += w
    wadd(rec.get("dose_value") is not None, 3)
    wadd("administration_route" in rec and rec.get("administration_route"), 2)
    wadd("tumor_inhibition_pct" in rec and rec.get("tumor_inhibition_pct") is not None, 2)
    wadd("particle_size_nm" in rec and rec.get("particle_size_nm") is not None, 1)
    wadd("tumor_types" in rec and rec.get("tumor_types"), 3)
    wadd("synthesis_method" in rec and rec.get("synthesis_method"), 1)
    wadd("synthesis_description" in rec and rec.get("synthesis_description"), 1)
    if rec.get("rag_used"): wadd(True, 1)
    rec["confidence"] = round((score / max(1.0, weight)) * 100.0, 1)
    if rec["confidence"] < 55 or rec.get("paywalled"):
        rec["needs_manual_check"] = True

    if debug:
        keys = [k for k in ("dose_value","dose_unit","administration_route","tumor_inhibition_pct","particle_size_nm","tumor_types") if rec.get(k) not in (None,"",[])]
        logging.info(f"[REC] pmid={rec.get('pmid')} keys={keys} conf={rec.get('confidence')} FT_used={rec.get('fulltext_used_for_extraction')} src={rec.get('fulltext_source')}")
    return rec

# Visualization
def visualize_and_save(df, out_png):
    if plt is None or df is None or df.empty: return
    indicators = {
        "administration_route": "Administration Route",
        "dose_bucket": "Dose (bucket)",
        "treatment_duration_bucket": "Treatment Duration",
        "tumor_inhibition_bucket": "Tumor Inhibition",
        "intervention_type": "Intervention",
        "tumor_model": "Tumor Model",
        "synthesis_complexity": "Synthesis Complexity"
    }
    def dose_bucket(v, unit):
        if v is None: return "unknown"
        try:
            if unit == "mg/kg":
                if v <= 1: return "<=1 mg/kg"
                if v <= 5: return "1-5 mg/kg"
                if v <= 10: return "5-10 mg/kg"
                return ">10 mg/kg"
            if unit == "mg/g":
                if v <= 0.1: return "<=0.1 mg/g"
                if v <= 1: return "0.1-1 mg/g"
                return ">1 mg/g"
            if unit in ("ug/mL","mg/mL"):
                if v <= 10: return "<=10"
                if v <= 50: return "10-50"
                return ">50"
        except Exception:
            pass
        return "unknown"
    def duration_bucket(d):
        if d is None: return "unknown"
        try: d = float(d)
        except Exception: return "unknown"
        if d <= 7: return "<=7d"
        if d <= 21: return "8-21d"
        if d <= 60: return "22-60d"
        return ">60d"
    def tgi_bucket(v):
        if v is None: return "unknown"
        try: v = float(v)
        except Exception: return "unknown"
        if v >= 80: return ">=80%"
        if v >= 50: return "50-79%"
        if v >= 20: return "20-49%"
        return "<20%"

    rows=[]
    for _, r in df.iterrows():
        rows.append({
            "administration_route": r.get("administration_route") or "unknown",
            "dose_bucket": dose_bucket(r.get("dose_value"), r.get("dose_unit")),
            "treatment_duration_bucket": duration_bucket(r.get("treatment_duration_days") or r.get("treatment_duration") or None),
            "tumor_inhibition_bucket": tgi_bucket(r.get("tumor_inhibition_pct")),
            "intervention_type": (r.get("interventions")[0] if isinstance(r.get("interventions"), list) and r.get("interventions") else "unknown"),
            "tumor_model": (r.get("tumor_types")[0] if isinstance(r.get("tumor_types"), list) and r.get("tumor_types") else "unknown"),
            "synthesis_complexity": r.get("synthesis_complexity") or "unknown"
        })
    agg = pd.DataFrame(rows)
    if agg.empty or plt is None: return
    fig, axes = plt.subplots(2, 4, figsize=(22, 10)); axes = axes.flatten(); idx = 0
    for col, title in indicators.items():
        ax = axes[idx]; idx += 1
        counts = agg[col].value_counts()
        if counts.empty: ax.axis('off'); continue
        counts.sort_values(ascending=True).plot(kind='barh', ax=ax, color=plt.cm.tab20.colors)
        ax.set_title(title)
    for j in range(idx, len(axes)): axes[j].axis('off')
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close()

# CLI
def main():
    parser = argparse.ArgumentParser(description="RAG+LLM+Regex therapeutic parameter extractor (no fabrication).")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to verified literature CSV")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    parser.add_argument("--outdir", default=OUTPUT_DIR, help="Directory for outputs and visualization")
    parser.add_argument("--try_fulltext", action="store_true", default=True, help="Attempt to fetch fulltext (default: True)")
    parser.add_argument("--unpaywall_email", default=None, help="Email for Unpaywall API")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--use_transformer", action="store_true", help="Enable QA fallback")
    parser.add_argument("--transformer_model", default="distilbert-base-cased-distilled-squad", help="QA model path/name")
    parser.add_argument("--use_rag", action="store_true", help="Enable PubMedBERT+FAISS RAG retrieval")
    parser.add_argument("--embedder_model", default="/home/donaldtangai4s/Desktop/ROS/PubMedBert", help="PubMedBERT path/name for embeddings")
    parser.add_argument("--use_llm", action="store_true", help="Use local LLM (OpenAI-compatible) for structured extraction")
    parser.add_argument("--llm_base_url", default=os.environ.get("LLM_BASE_URL","http://127.0.0.1:8000/v1"))
    parser.add_argument("--llm_model", default=os.environ.get("LLM_MODEL","Qwen2.5-7B-Instruct"))
    parser.add_argument("--llm_api_key", default=os.environ.get("LLM_API_KEY"))
    parser.add_argument("--llm_max_tokens", type=int, default=600)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(FULLTEXT_DIR, exist_ok=True)

    if not os.path.exists(args.input):
        print("Input CSV not found:", args.input); return
    df = pd.read_csv(args.input, dtype=str)
    if df.empty:
        print("Input CSV empty:", args.input); return

    qa_pipe = None
    if args.use_transformer:
        if not _TRANSFORMERS_AVAILABLE:
            print("Transformers not available."); return
        try:
            model_to_load = args.transformer_model
            local_model_dir = "/home/donaldtangai4s/Desktop/ROS/PubMedBert"
            if os.path.isdir(local_model_dir): model_to_load = local_model_dir
            qa_pipe = init_qa_pipeline(model_name=model_to_load, device=(0 if torch and torch.cuda.is_available() else -1), local_only=True)
            if args.debug: print("QA pipeline initialized:", model_to_load)
        except Exception as e:
            print("Warning: QA init failed:", e); qa_pipe=None; args.use_transformer=False

    llm_cfg = None
    if args.use_llm:
        llm_cfg = dict(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            max_tokens=args.llm_max_tokens
        )

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for _, row in df.iterrows():
            futures[ex.submit(
                process_row,
                row.to_dict(),
                args.unpaywall_email,
                args.try_fulltext,
                args.use_transformer,
                qa_pipe,
                20,
                args.debug,
                args.use_rag,
                args.embedder_model,
                llm_cfg if args.use_llm else None
            )] = row.get("pmid")
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"pmid": futures[fut], "status": "error", "needs_manual_check": True})
                if args.debug: logging.error(f"[ROW][ERR] pmid={futures[fut]} err={e}")

    out_df = pd.DataFrame(results)
    merged = df.merge(out_df, on="pmid", how="right", suffixes=("", "_extracted"))
    merged.to_csv(args.output, index=False)
    print("Saved extracted CSV to:", args.output)

    try:
        visualize_and_save(merged, os.path.join(args.outdir, "parameter_distribution_stacked_RAG.png"))
        print("Saved visualization to:", os.path.join(args.outdir, "parameter_distribution_stacked_RAG.png"))
    except Exception as e:
        print("Visualization failed:", e)

if __name__ == "__main__":
    import numpy as np
    main()