import os
import pandas as pd
import re
from collections import Counter
import PyPDF2
import numpy as np

CSV_FILE = "/home/donaldtangai4s/Desktop/ROS/new_query/pubmed_results_2015_2025.csv"
FULLTEXT_DIR = "/home/donaldtangai4s/Desktop/ROS/new_query/FullText"

def extract_text_from_pdf(pdf_path):
    """从PDF文件中提取文本，分别返回摘要和正文"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
        
        # 尝试分离摘要和正文
        abstract_text = extract_abstract(full_text)
        return full_text, abstract_text
    except Exception as e:
        print(f"提取PDF文本失败 {pdf_path}: {e}")
        return "", ""

def extract_abstract(text):
    """提取摘要部分"""
    abstract_patterns = [
        r'Abstract[:\s]*(.*?)(?:Keywords|Introduction|1\.|INTRODUCTION|Materials|Methods)',
        r'ABSTRACT[:\s]*(.*?)(?:KEYWORDS|INTRODUCTION|1\.|Introduction|MATERIALS|METHODS)',
        r'Summary[:\s]*(.*?)(?:Keywords|Introduction|1\.|INTRODUCTION)',
    ]
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # 限制摘要长度，避免包含过多内容
            words = abstract.split()
            if len(words) > 300:
                abstract = ' '.join(words[:300])
            return abstract
    
    # 如果没找到明确的摘要标记，取前400个词作为摘要
    words = text.split()
    if len(words) > 400:
        return ' '.join(words[:400])
    return text[:2000]  # 限制在2000字符内

def check_keyword_frequency(text, keywords, min_frequency=2):
    """检查关键词在正文中的出现频率"""
    text_lower = text.lower()
    valid_keywords = []
    
    for keyword in keywords:
        if isinstance(keyword, str):
            # 使用单词边界匹配，避免部分匹配
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            count = len(re.findall(pattern, text_lower))
            if count >= min_frequency:
                valid_keywords.append(keyword)
    
    return valid_keywords

# 1. 从摘要中提取核心参数

def extract_nanomaterial_type(text):
    """从文本中提取纳米材料类型"""
    # 确保输入是字符串
    if isinstance(text, tuple):
        text = text[0] if text[0] else text[1] if len(text) > 1 else ""
    
    if not isinstance(text, str) or not text:
        return []
    
    nano_types = {
        'nanoparticle': r'nanoparticle[s]?|NP[s]?(?!\w)',
        'nanorod': r'nanorod[s]?|nanowire[s]?',
        'nanosheet': r'nanosheet[s]?|nanoplate[s]?',
        'nanotube': r'nanotube[s]?|CNT[s]?',
        'nanosphere': r'nanosphere[s]?|nanobead[s]?',
        'nanocrystal': r'nanocrystal[s]?|NC[s]?'
    }
    
    found_types = []
    text_lower = text.lower()
    
    for nano_type, pattern in nano_types.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            found_types.append(nano_type)
    
    return found_types
def extract_nanomaterial_name(text):
    """从文本中提取纳米材料名称"""
    # 确保输入是字符串
    if isinstance(text, tuple):
        text = text[0] if text[0] else text[1] if len(text) > 1 else ""
    
    if not isinstance(text, str) or not text:
        return []
    
    materials = {
        'TiO2': r'TiO2|titanium\s+dioxide|titania',
        'MnO2': r'MnO2|manganese\s+dioxide',
        'MgO2': r'MgO2|magnesium\s+dioxide',
        'ZnO': r'ZnO|zinc\s+oxide',
        'Fe3O4': r'Fe3O4|magnetite|iron\s+oxide',
        'CuO': r'CuO|copper\s+oxide',
        'SiO2': r'SiO2|silica|silicon\s+dioxide',
        'CeO2': r'CeO2|cerium\s+oxide|ceria',
        'Cu2O': r'Cu2O|cuprous\s+oxide',
        'SnO2': r'SnO2|tin\s+oxide',
        'WO3': r'WO3|tungsten\s+oxide',
        'Nb2O5': r'Nb2O5|niobium\s+oxide',
        'ZrO2': r'ZrO2|zirconia|zirconium\s+oxide',
        'Al2O3': r'Al2O3|alumina|aluminum\s+oxide',
        'Bi2WO6': r'Bi2WO6|bismuth\s+tungstate',
        'BiVO4': r'BiVO4|bismuth\s+vanadate',
        'CeO2': r'CeO2|cerium\s+oxide|ceria',
        'Fe2O3': r'Fe2O3|hematite|iron\s+oxide',
        'Cr2O3': r'Cr2O3|chromium\s+oxide',
        'Co3O4': r'Co3O4|cobalt\s+oxide',
        'V2O5': r'V2O5|vanadium\s+oxide',
        'MoO3': r'MoO3|molybdenum\s+oxide',
        'BaTiO3': r'BaTiO3|barium\s+titanate',
        'SrTiO3': r'SrTiO3|strontium\s+titanate',
        #sulfide oxide
        'ZnS': r'ZnS|zinc\s+sulfide',
        'CdS': r'CdS|cadmium\s+sulfide',
        'MoS2': r'MoS2|molybdenum\s+sulfide',
        'WS2': r'WS2|tungsten\s+sulfide',
        'CuS': r'CuS|copper\s+sulfide',
        'FeS2': r'FeS2|pyrite|iron\s+sulfide',
        'Ag2S': r'Ag2S|silver\s+sulfide',
        'AuS': r'AuS|gold\s+sulfide',
        'PtS': r'PtS|platinum\s+sulfide',
        'RuS': r'RuS|ruthenium\s+sulfide',
        #chloride oxide
        'TiCl3': r'TiCl3|titanium\s+chloride',
        'FeCl3': r'FeCl3|ferric\s+chloride|iron\s+chloride',
        'CuCl2': r'CuCl2|cupric\s+chloride|copper\s+chloride',
        'ZnCl2': r'ZnCl2|zinc\s+chloride',
        'SnCl2': r'SnCl2|stannous\s+chloride|tin\s+chloride',
        'AlCl3': r'AlCl3|aluminum\s+chloride',
        'PCl3': r'PCl3|phosphorus\s+chloride',
        'NaCl': r'NaCl|sodium\s+chloride',
        'KCl': r'KCl|potassium\s+chloride',
        #piezoelectric materials
        'BaTiO3': r'BaTiO3|barium\s+titanate',
        'PZT': r'PZT|lead\s+zirconate\s+titanate',
        'ZnO': r'ZnO|zinc\s+oxide',
        'PVDF': r'PVDF|polyvinylidene\s+fluoride',
        'KNN': r'KNN|potassium\s+sodium\s+niobate',
        'LiNbO3': r'LiNbO3|lithium\s+niobate',
        'LiTaO3': r'LiTaO3|lithium\s+tantalate',
        'AlN': r'AlN|aluminum\s+nitride'

    }
    
    found_materials = []
    for material, pattern in materials.items():
        if re.search(pattern, text, re.IGNORECASE):
            found_materials.append(material)
    
    return found_materials

def extract_tumor_type(text):
    """从文本中提取肿瘤种类"""
    # 确保输入是字符串
    if isinstance(text, tuple):
        text = text[0] if text[0] else text[1] if len(text) > 1 else ""
    
    if not isinstance(text, str) or not text:
        return []
    
    tumor_types = {
        'breast_cancer': r'breast\s+cancer|mammary\s+tumor|mammary\s+carcinoma|4T1',
        'lung_cancer': r'lung\s+cancer|pulmonary\s+tumor|lung\s+carcinoma|A549|H460',
        'liver_cancer': r'liver\s+cancer|hepatocellular\s+carcinoma|HCC|hepatoma|HepG2',
        'colon_cancer': r'colon\s+cancer|colorectal\s+cancer|rectal\s+cancer|HCT116|SW480',
        'prostate_cancer': r'prostate\s+cancer|prostate\s+carcinoma|PC-3|DU145',
        'melanoma': r'melanoma|skin\s+cancer|B16|B16@Ova',
        'glioma': r'glioma|brain\s+tumor|glioblastoma|U87|C6',
        'pancreatic_cancer': r'pancreatic\s+cancer|pancreatic\s+carcinoma|PANC-1',
        'ovarian_cancer': r'ovarian\s+cancer|ovarian\s+carcinoma|SKOV3',
        'cervical_cancer': r'cervical\s+cancer|cervical\s+carcinoma|HeLa|TC-1',
        'gastric_cancer': r'gastric\s+cancer|stomach\s+cancer|SGC-7901',
        'bladder_cancer': r'bladder\s+cancer|bladder\s+carcinoma|T24',
        'kidney_cancer': r'kidney\s+cancer|renal\s+cancer|renal\s+carcinoma',
        'thyroid_cancer': r'thyroid\s+cancer|thyroid\s+carcinoma',
        'esophageal_cancer': r'esophageal\s+cancer|esophageal\s+carcinoma',
        'head_neck_cancer': r'head\s+and\s+neck\s+cancer|HNSCC',
        'sarcoma': r'sarcoma|osteosarcoma|U2OS',
        'leukemia': r'leukemia|lymphoma'
    }
    
    found_tumors = []
    for tumor, pattern in tumor_types.items():
        if re.search(pattern, text, re.IGNORECASE):
            found_tumors.append(tumor)
    
    return found_tumors

def extract_intervention_methods(abstract):
    """从摘要中提取干涉方法"""
    intervention_methods = {
        'sonodynamic': r'sonodynamic\s+therapy|SDT|sonogenetic\s|ultrasound|US|focused ultrasound|FUS|HIFU|acoustic',
        'photodynamic': r'photodynamic\s+therapy|PDT|laser|light\s+irradiation|phototherapy',
        'microwave': r'microwave|MW|microwave\s+therapy',
        'photothermal': r'photothermal\s+therapy|PTT|NIR|near-infrared|infrared|thermal',
        'magnetic': r'magnetic|magnetic\s+resonance|MRI|magnetic\s+imaging|MRI|magnetic resonance imaging|PET|PET|positron\s+emission\s+tomography|PET-CT',
        'radiotherapy': r'radiofrequency|RF|radio.frequency|electron\s+beam|gamma\s+ray|X-ray\s+therapy|XRT',
        'chemotherapy': r'chemotherapy|chemo|drug\s+therapy|drug\s+delivery|pharmacotherapy|pharmacodynamics|pharmacokinetics|Doxorubicin|Cisplatin|Paclitaxel|5-FU|5fluorouracil|Gemcitabine|Temozolomide|TMZ',
        'immunotherapy': r'immunotherapy|immuno|immune\s+therapy|immune\s+checkpoint|PD-1|PD-L1|CTLA-4|CAR-T|vaccine|adjuvant',
        'bacterotherapy': r'bacterial\s+therapy|bacteria|escherichia\s+coli|salmonella|clostridium|listeria|virus|bacteriophage|oncolytic\s+virus|phage'
    }
    
    found_methods = []
    abstract_lower = abstract.lower()
    
    for method, pattern in intervention_methods.items():
        if re.search(pattern, abstract_lower, re.IGNORECASE):
            found_methods.append(method)
    
    return found_methods

def extract_tumor_type(abstract):
    """从摘要中提取肿瘤种类"""
    tumor_types = {
        'breast_cancer': r'breast\s+cancer|mammary\s+tumor|mammary\s+carcinoma|MCF-7|MDA-MB',
        'lung_cancer': r'lung\s+cancer|pulmonary\s+tumor|lung\s+carcinoma|A549|H460',
        'liver_cancer': r'liver\s+cancer|hepatocellular\s+carcinoma|HCC|hepatoma|HepG2',
        'colon_cancer': r'colon\s+cancer|colorectal\s+cancer|rectal\s+cancer|HCT116|SW480',
        'prostate_cancer': r'prostate\s+cancer|prostate\s+carcinoma|PC-3|DU145',
        'melanoma': r'melanoma|skin\s+cancer|B16|A375',
        'glioma': r'glioma|brain\s+tumor|glioblastoma|U87|C6',
        'pancreatic_cancer': r'pancreatic\s+cancer|pancreatic\s+carcinoma|PANC-1',
        'ovarian_cancer': r'ovarian\s+cancer|ovarian\s+carcinoma|SKOV3',
        'cervical_cancer': r'cervical\s+cancer|cervical\s+carcinoma|HeLa',
        'gastric_cancer': r'gastric\s+cancer|stomach\s+cancer|SGC-7901',
        'bladder_cancer': r'bladder\s+cancer|bladder\s+carcinoma|T24',
        'kidney_cancer': r'kidney\s+cancer|renal\s+cancer|renal\s+carcinoma',
        'thyroid_cancer': r'thyroid\s+cancer|thyroid\s+carcinoma',
        'esophageal_cancer': r'esophageal\s+cancer|esophageal\s+carcinoma',
        'head_neck_cancer': r'head\s+and\s+neck\s+cancer|HNSCC',
        'sarcoma': r'sarcoma|osteosarcoma|U2OS',
        'leukemia': r'leukemia|lymphoma'
    }
    
    found_tumors = []
    for tumor, pattern in tumor_types.items():
        if re.search(pattern, abstract, re.IGNORECASE):
            found_tumors.append(tumor)
    
    return found_tumors

def extract_particle_size(text):
    """提取粒径信息"""
    size_patterns = [
        r'diameter[:\s]*(\d+(?:\.\d+)?)\s*nm',
        r'DLS[:\s]*(\d+(?:\.\d+)?)\s*nm',
        r'size[:\s]*(\d+(?:\.\d+)?)\s*nm',
        r'(\d+(?:\.\d+)?)\s*nanometer[s]?',
        r'particle\s+size[:\s]*(\d+(?:\.\d+)?)\s*nm'
    ]
    
    sizes = []
    for pattern in size_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        sizes.extend([float(match) for match in matches])
    
    return sizes

def extract_zeta_potential(text):
    """提取Zeta电位"""
    zeta_patterns = [
        r'zeta\s+potential[:\s]*([+-]?\d+(?:\.\d+)?)\s*mV',
        r'zeta[:\s]*([+-]?\d+(?:\.\d+)?)\s*mV',
        r'ζ[:\s]*([+-]?\d+(?:\.\d+)?)\s*mV',
        r'surface\s+charge[:\s]*([+-]?\d+(?:\.\d+)?)\s*mV'
    ]
    
    potentials = []
    for pattern in zeta_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        potentials.extend([float(match) for match in matches])
    
    return potentials

def extract_pzc(text):
    """提取零电荷点(Point of Zero Charge, PZC)"""
    pzc_patterns = [
        r'PZC[:\s]*(\d+(?:\.\d+)?)',
        r'point\s+of\s+zero\s+charge[:\s]*(\d+(?:\.\d+)?)',
        r'isoelectric\s+point[:\s]*(\d+(?:\.\d+)?)',
        r'IEP[:\s]*(\d+(?:\.\d+)?)'
    ]
    
    pzc_values = []
    for pattern in pzc_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        pzc_values.extend([float(match) for match in matches])
    
    return pzc_values

"""提取注射方法，只提取出现两次以上的方法"""
def extract_injection_method(text):
    """
    只提取在全文中出现次数>=2的注射方法
    """
    injection_methods = {
        'intravenous': r'intravenous|IV|tail\s+vein',
        'intraperitoneal': r'intraperitoneal|IP',
        'intratumoral': r'intratumoral|IT',
        'oral': r'oral|gavage',
        'subcutaneous': r'subcutaneous|SC',
        'intramuscular': r'intramuscular|IM',
        'intracardiac': r'intracardiac|IC',
        'intracerebral': r'intracerebral|intracranial', 
        'intradermal': r'intradermal|ID'
    }
    found_methods = []
    for method, pattern in injection_methods.items():
        count = len(re.findall(pattern, text, re.IGNORECASE))
        if count >= 3:
            found_methods.append(method)
    return found_methods

def extract_injection_dose(text):
    """提取注射剂量"""
    dose_patterns = [
        r'(\d+(?:\.\d+)?)\s*mg/kg',
        r'dose[:\s]*(\d+(?:\.\d+)?)\s*mg/kg',
        r'(\d+(?:\.\d+)?)\s*μg/kg',
        r'(\d+(?:\.\d+)?)\s*mg\s*kg[-]?1',
        r'concentration[:\s]*(\d+(?:\.\d+)?)\s*mg/mL'
    ]
    
    doses = []
    for pattern in dose_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        doses.extend([float(match) for match in matches])
    
    return doses

def extract_injection_frequency(text):
    """提取注射次数/频率"""
    freq_patterns = [
        r'(\d+)\s*injection[s]?',
        r'injected\s*(\d+)\s*time[s]?',
        r'administered\s*(\d+)\s*time[s]?',
        r'once\s*daily|daily',
        r'twice\s*daily',
        r'every\s*(\d+)\s*day[s]?'
    ]
    
    frequencies = []
    for pattern in freq_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], str) and matches[0].isdigit():
                frequencies.append(int(matches[0]))
    
    return frequencies

def extract_ultrasound_parameters(text):
    """提取超声参数"""
    us_params = {}
    
    # 频率
    freq_patterns = [
        r'(\d+(?:\.\d+)?)\s*MHz',
        r'frequency[:\s]*(\d+(?:\.\d+)?)\s*MHz'
    ]
    frequencies = []
    for pattern in freq_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        frequencies.extend([float(match) for match in matches])
    us_params['frequency_MHz'] = frequencies
    
    # 功率
    power_patterns = [
        r'(\d+(?:\.\d+)?)\s*W/cm2',
        r'power[:\s]*(\d+(?:\.\d+)?)\s*W/cm2',
        r'intensity[:\s]*(\d+(?:\.\d+)?)\s*W/cm2'
    ]
    powers = []
    for pattern in power_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        powers.extend([float(match) for match in matches])
    us_params['power_W_cm2'] = powers

    # 占空比
    duty_patterns = [
        r'(\d+(?:\.\d+)?)\s*%?\s*duty\s*cycle',
        r'duty\s*cycle[:\s]*(\d+(?:\.\d+)?)\s*%?'
    ]
    duties = []
    for pattern in duty_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        duties.extend([float(match) for match in matches])
    us_params['duty_cycle_%'] = duties  

    # 治疗时间
    time_patterns = [
        r'(\d+)\s*min[ute[s]?]?',
        r'treatment\s+time[:\s]*(\d+)\s*min',
        r'sonication[:\s]*(\d+)\s*min'
    ]
    times = []
    for pattern in time_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        times.extend([int(match) for match in matches])
    us_params['treatment_time_min'] = times
    
    # 治疗次数
    session_patterns = [
        r'(\d+)\s*session[s]?',
        r'treated\s*(\d+)\s*time[s]?',
        r'(\d+)\s*treatment[s]?'
    ]
    sessions = []
    for pattern in session_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        sessions.extend([int(match) for match in matches])
    us_params['treatment_sessions'] = sessions
    
    return us_params

def extract_ros_generation(text):
    """提取ROS生成量"""
    ros_patterns = [
        r'ROS\s+generation[:\s]*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*μM\s*min[-]?1',
        r'singlet\s+oxygen[:\s]*(\d+(?:\.\d+)?)',
        r'1O2[:\s]*(\d+(?:\.\d+)?)',
        r'hydroxyl\s+radical[:\s]*(\d+(?:\.\d+)?)',
        #羟基自由基
        r'·OH[:\s]*(\d+(?:\.\d+)?)',
        #超氧阴离子
        r'O2·-[:\s]*(\d+(?:\.\d+)?)'
        #各种探针
        r'DCFH-DA[:\s]*(\d+(?:\.\d+)?)',
        r'ABDA[:\s]*(\d+(?:\.\d+)?)',
        r'DPBF[:\s]*(\d+(?:\.\d+)?)',
        r'ESR[:\s]*(\d+(?:\.\d+)?)',
        r'TMB[:\s]*(\d+(?:\.\d+)?)',
        r'NBT[:\s]*(\d+(?:\.\d+)?)',
        r'MB[:\s]*(\d+(?:\.\d+)?)',
        r'RhB[:\s]*(\d+(?:\.\d+)?)'
    ]
    
    ros_values = []
    for pattern in ros_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        ros_values.extend([float(match) for match in matches])
    
    return ros_values

def extract_cck8_viability(text):
    """提取CCK8细胞存活率"""
    cck8_patterns = [
        r'CCK.?8[:\s]*(\d+(?:\.\d+)?)\s*%',
        r'DAPI[:\s]*(\d+(?:\.\d+)?)\s*%',
        r"Holchest[:\s]*(\d+(?:\.\d+)?)\s*%",
        r'MTT[:\s]*(\d+(?:\.\d+)?)\s*%',
        r'cell\s+survival[:\s]*(\d+(?:\.\d+)?)\s*%',
        r'cell\s+viability[:\s]*(\d+(?:\.\d+)?)\s*%',
        r'survival\s+rate[:\s]*(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s*viability'
    ]
    
    viabilities = []
    for pattern in cck8_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        viabilities.extend([float(match) for match in matches])
    
    return viabilities

def extract_tumor_volume(text):
    """提取肿瘤体积"""
    volume_patterns = [
        r'tumor\s+volume[:\s]*(\d+(?:\.\d+)?)\s*mm3',
        r'volume[:\s]*(\d+(?:\.\d+)?)\s*mm3',
        r'(\d+(?:\.\d+)?)\s*mm3',
        r'tumor\s+size[:\s]*(\d+(?:\.\d+)?)\s*cm3'
    ]
    
    volumes = []
    for pattern in volume_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        volumes.extend([float(match) for match in matches])
    
def extract_sdt_features():
    """提取SDT相关特征"""
    print("开始提取声动力学治疗(SDT)相关特征...")
    
    # 获取PDF文件列表
    pdf_files = [f for f in os.listdir(FULLTEXT_DIR) if f.lower().endswith('.txt')]
    print(f"找到 {len(pdf_files)} 个篇；")
    
    # 创建特征DataFrame
    features_data = []
    
    for i, pdf_file in enumerate(pdf_files[:262]):  # 262处理个文件
        print(f"处理文件 {i+1}/{min(262, len(pdf_files))}: {pdf_file}")
        
        pdf_path = os.path.join(FULLTEXT_DIR, pdf_file)
        
        # 提取PDF文本 - 修复这里，正确解包元组
        full_text, abstract_text = extract_text_from_pdf(pdf_path)
        
        if full_text:
            # 优先使用摘要进行特征提取，如果没有摘要则使用全文
            text_for_extraction = abstract_text if abstract_text else full_text
            
            # 提取所有SDT相关特征
            nano_types = extract_nanomaterial_type(text_for_extraction)
            nano_materials = extract_nanomaterial_name(text_for_extraction)
            tumor_types = extract_tumor_type(text_for_extraction)
            
            # 从全文中提取数值参数（这些通常在正文中）
            particle_sizes = extract_particle_size(full_text)
            zeta_potentials = extract_zeta_potential(full_text)
            pzc_values = extract_pzc(full_text)
            injection_methods = extract_injection_method(full_text)
            injection_doses = extract_injection_dose(full_text)
            injection_frequencies = extract_injection_frequency(full_text)
            us_params = extract_ultrasound_parameters(full_text)
            ros_values = extract_ros_generation(full_text)
            cck8_values = extract_cck8_viability(full_text)
            tumor_volumes = extract_tumor_volume(full_text)
            
            # 处理数据，取平均值或第一个值
            def safe_mean(values):
                return np.mean(values) if values else None
            
            def safe_first(values):
                return values[0] if values else None
            
            def safe_join(values):
                return ', '.join(values) if values else None
            
            # 构建特征行
            feature_row = {
                'pdf_filename': pdf_file,
                
                # 纳米材料特征
                'nanomaterial_types': safe_join(nano_types),
                'nanomaterial_names': safe_join(nano_materials),
                'primary_nanomaterial': safe_first(nano_materials),
                'primary_nano_type': safe_first(nano_types),
                
                # 物理化学特征
                'particle_size_nm': safe_mean(particle_sizes),
                'particle_size_range': f"{min(particle_sizes)}-{max(particle_sizes)}" if len(particle_sizes) > 1 else safe_first(particle_sizes),
                'zeta_potential_mV': safe_mean(zeta_potentials),
                'pzc_value': safe_mean(pzc_values),
                
                # 生物学特征
                'tumor_types': safe_join(tumor_types),
                'primary_tumor_type': safe_first(tumor_types),
                
                # 给药特征
                'injection_methods': safe_join(injection_methods),
                'primary_injection_method': safe_first(injection_methods),
                'injection_dose_mg_kg': safe_mean(injection_doses),
                'injection_frequency': safe_mean(injection_frequencies),
                
                # 超声治疗参数
                'ultrasound_frequency_MHz': safe_mean(us_params.get('frequency_MHz', [])),
                'ultrasound_power_W_cm2': safe_mean(us_params.get('power_W_cm2', [])),
                'treatment_time_min': safe_mean(us_params.get('treatment_time_min', [])),
                'treatment_sessions': safe_mean(us_params.get('treatment_sessions', [])),
                
                # 效果评估
                'ros_generation': safe_mean(ros_values),
                'cell_viability_percent': safe_mean(cck8_values),
                'cell_death_percent': 100 - safe_mean(cck8_values) if cck8_values else None,
                'tumor_volume_mm3': safe_mean(tumor_volumes),
                
                # 综合评估指标
                'has_size_data': 'yes' if particle_sizes else 'no',
                'has_zeta_data': 'yes' if zeta_potentials else 'no',
                'has_ultrasound_data': 'yes' if any(us_params.values()) else 'no',
                'has_ros_data': 'yes' if ros_values else 'no',
                'has_viability_data': 'yes' if cck8_values else 'no',
                'has_volume_data': 'yes' if tumor_volumes else 'no',
                
                # 数据完整性评分 (0-6)
                'data_completeness_score': sum([
                    1 if particle_sizes else 0,
                    1 if zeta_potentials else 0,
                    1 if any(us_params.values()) else 0,
                    1 if ros_values else 0,
                    1 if cck8_values else 0,
                    1 if tumor_volumes else 0
                ])
            }
            
            features_data.append(feature_row)
        else:
            print(f"  无法提取文本: {pdf_file}")
    
    # 创建特征DataFrame
    df_features = pd.DataFrame(features_data)
    
    if not df_features.empty:
        print(f"\n=== 成功提取 {len(df_features)} 个文件的SDT特征 ===")
        
        # 显示纳米材料统计
        print("\n=== 纳米材料类型分布 ===")
        nano_type_counts = {}
        for types in df_features['nanomaterial_types'].dropna():
            for nano_type in types.split(', '):
                nano_type_counts[nano_type] = nano_type_counts.get(nano_type, 0) + 1
        for nano_type, count in sorted(nano_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {nano_type}: {count}")
        
        print("\n=== 纳米材料名称分布 ===")
        material_counts = {}
        for materials in df_features['nanomaterial_names'].dropna():
            for material in materials.split(', '):
                material_counts[material] = material_counts.get(material, 0) + 1
        for material, count in sorted(material_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {material}: {count}")
        
        print("\n=== 肿瘤类型分布 ===")
        tumor_counts = {}
        for tumors in df_features['tumor_types'].dropna():
            for tumor in tumors.split(', '):
                tumor_counts[tumor] = tumor_counts.get(tumor, 0) + 1
        for tumor, count in sorted(tumor_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tumor}: {count}")
        
        # 显示数值特征统计
        print("\n=== 关键数值特征统计 ===")
        numeric_cols = ['particle_size_nm', 'zeta_potential_mV', 'injection_dose_mg_kg', 
                       'ultrasound_frequency_MHz', 'ultrasound_power_W_cm2', 'treatment_time_min',
                       'ros_generation', 'cell_viability_percent', 'tumor_volume_mm3']
        
        for col in numeric_cols:
            if col in df_features.columns:
                values = df_features[col].dropna()
                if len(values) > 0:
                    print(f"\n{col}:")
                    print(f"  样本数: {len(values)}")
                    print(f"  平均值: {values.mean():.2f}")
                    print(f"  范围: {values.min():.2f} - {values.max():.2f}")
        
        # 显示数据完整性
        print("\n=== 数据完整性分析 ===")
        completeness_dist = df_features['data_completeness_score'].value_counts().sort_index()
        for score, count in completeness_dist.items():
            print(f"  完整性评分 {score}/6: {count} 篇文献")
        
        # 显示最完整的数据
        print(f"\n=== 数据最完整的前5篇文献 ===")
        top_complete = df_features.nlargest(5, 'data_completeness_score')
        for idx, row in top_complete.iterrows():
            print(f"\n文件: {row['pdf_filename']}")
            print(f"  完整性评分: {row['data_completeness_score']}/6")
            print(f"  纳米材料: {row['primary_nanomaterial']}")
            print(f"  粒径: {row['particle_size_nm']} nm")
            print(f"  Zeta电位: {row['zeta_potential_mV']} mV")
            print(f"  肿瘤类型: {row['primary_tumor_type']}")
            print(f"  细胞存活率: {row['cell_viability_percent']}%")
        
        # 保存特征数据
        output_file = "/home/donaldtangai4s/Desktop/ROS/new_query/sdt_features.csv"
        df_features.to_csv(output_file, index=False)
        print(f"\nSDT特征数据已保存到: {output_file}")
        
        # 创建用于机器学习的简化数据集
        ml_features = df_features[[
            'pdf_filename', 'primary_nanomaterial', 'primary_nano_type',
            'particle_size_nm', 'zeta_potential_mV', 'injection_dose_mg_kg',
            'ultrasound_frequency_MHz', 'ultrasound_power_W_cm2', 'treatment_time_min',
            'cell_death_percent', 'tumor_volume_mm3', 'data_completeness_score'
        ]].dropna(subset=['primary_nanomaterial'])
        
        if not ml_features.empty:
            ml_output_file = "/home/donaldtangai4s/Desktop/ROS/new_query/sdt_ml_features.csv"
            ml_features.to_csv(ml_output_file, index=False)
            print(f"机器学习特征数据已保存到: {ml_output_file}")
            print(f"机器学习数据集包含 {len(ml_features)} 个样本")
    else:
        print("未能提取任何SDT特征数据")
    
    return df_features

if __name__ == "__main__":
    sdt_features_df = extract_sdt_features()