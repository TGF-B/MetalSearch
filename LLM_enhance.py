#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
import re
import time  # Added for sleep
from PyPDF2 import PdfReader
import ollama  # Requires: pip install ollama
from typing import Dict, Any
import numpy as np  # For NaN handling

# ========= Configuration =========
FEATURES_CSV = "/home/donaldtangai4s/Desktop/ROS/new_query/sdt_features.csv"
TEXT_DIR = "/home/donaldtangai4s/Desktop/ROS/new_query/Text"
OUTPUT_CSV = "/home/donaldtangai4s/Desktop/ROS/new_query/sdt_features_corrected.csv"
OLLAMA_MODEL = "qwen2.5:7b"  # Assume your QWen model name; change if different
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama host
BATCH_SIZE = 5  # Process 5 papers per batch to avoid timeouts; adjustable
SLEEP_TIME = 2  # Delay between model calls

# Initialize Ollama client
client = ollama.Client(host=OLLAMA_HOST)

def generate_correction_prompt(original_features: Dict[str, Any], full_text: str) -> str:
    """Generate Ollama prompt for contextual semantic correction and table filling"""
    # Key features list (based on original script, matching CSV columns)
    key_features = [
        "primary_nanomaterial", "primary_nano_type", "particle_size_nm", "zeta_potential_mV",
        "pzc_value", "primary_tumor_type", "primary_injection_method", "injection_dose_mg_kg",
        "injection_frequency", "ultrasound_frequency_MHz", "ultrasound_power_W_cm2",
        "treatment_time_min", "treatment_sessions", "ros_generation", "cell_viability_percent",
        "cell_death_percent", "tumor_volume_mm3"
    ]
    
    # Build original features string, handling NaN/None
    orig_str_parts = []
    for k, v in original_features.items():
        if k in key_features:
            if pd.isna(v) or v is None:
                v_str = "NA"
            else:
                v_str = str(v)
            orig_str_parts.append(f"{k}: {v_str}")
    orig_str = "\n".join(orig_str_parts)
    
    prompt = f"""
You are an expert in nanomaterial-ultrasound cancer therapy. Based on the following full paper text (abstract + body), correct and fill the extracted original features.
Original features (may have errors, missing, or inaccuracies):
{orig_str}

Full paper text:
{full_text}

Task:
1. **Contextual Semantic Correction**: Review the full text and correct errors in the original features (e.g., inaccurate size units, misjudged material types). Keep original values if reasonable; otherwise, infer corrections from context.
2. **Table Filling**: For missing features (NA), extract from text or reasonably infer (e.g., estimate size based on similar materials). Use "NA" if undetermined.
3. **Output Strict JSON**: Output only the following JSON object, no other text. Keep numbers as float/int, strings concise.
{{
    "corrected_primary_nanomaterial": "str (e.g., 'TiO2') or NA",
    "corrected_primary_nano_type": "str (e.g., 'nanoparticle') or NA",
    "corrected_particle_size_nm": "float or NA",
    "corrected_zeta_potential_mV": "float or NA",
    "corrected_pzc_value": "float or NA",
    "corrected_primary_tumor_type": "str (e.g., 'breast_cancer') or NA",
    "corrected_primary_injection_method": "str (e.g., 'intravenous') or NA",
    "corrected_injection_dose_mg_kg": "float or NA",
    "corrected_injection_frequency": "int or NA",
    "corrected_ultrasound_frequency_MHz": "float or NA",
    "corrected_ultrasound_power_W_cm2": "float or NA",
    "corrected_treatment_time_min": "int or NA",
    "corrected_treatment_sessions": "int or NA",
    "corrected_ros_generation": "float or NA",
    "corrected_cell_viability_percent": "float or NA",
    "corrected_cell_death_percent": "float or NA",
    "corrected_tumor_volume_mm3": "float or NA",
    "correction_notes": "str (brief notes on key corrections, e.g., 'Adjusted size from methods section context') or ''"
}}

Ensure the output is valid JSON.
"""
    return prompt

def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> Dict[str, Any]:
    """Call Ollama QWen model"""
    try:
        print("Calling Ollama...")
        response = client.generate(model=model, prompt=prompt, options={"temperature": 0.3, "top_p": 0.9})
        print(f"Raw response: {response['response'][:200]}...")  # Debug: Print first 200 chars
        
        # Parse response (assume model outputs JSON)
        try:
            corrected = json.loads(response['response'])
            print("JSON parsed successfully")
            return corrected
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print("Attempting regex extraction...")
            # Fallback: Extract JSON block with regex, more robust
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response['response'], re.DOTALL)
            if json_match:
                try:
                    corrected = json.loads(json_match.group())
                    print("Regex JSON parsed successfully")
                    return corrected
                except json.JSONDecodeError:
                    print("Regex extraction also failed")
            return {}
    except Exception as e:
        print(f"Ollama call failed: {e}")
        return {}

def extract_text_from_txt(txt_path: str) -> str:
    """Directly read TXT text"""
    if not os.path.exists(txt_path):
        print(f"TXT file not found: {txt_path}")
        return ""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"Loaded TXT: {len(content)} chars")  # Debug
            return content[:10000]  # Limit length to prevent LLM overload
    except Exception as e:
        print(f"Failed to read TXT {txt_path}: {e}")
        return ""

def correct_features_batch(df_batch: pd.DataFrame) -> pd.DataFrame:
    """Correct a batch of records"""
    corrected_batch = []
    for idx, row in df_batch.iterrows():
        pdf_filename = row['pdf_filename']
        txt_filename = os.path.splitext(pdf_filename)[0] + ".txt"
        txt_path = os.path.join(TEXT_DIR, txt_filename)
        
        print(f"Correcting {txt_filename} ...")
        print(f"TXT path: {txt_path}")
        
        full_text = extract_text_from_txt(txt_path)
        if not full_text:
            print(f"  Skipping {txt_filename}: No text or file missing")
            # Still add original row with note
            corrected_row = row.copy()
            corrected_row["correction_notes"] = "Skipped: No text file"
            corrected_batch.append(corrected_row)
            continue
        
        original_features = row.to_dict()
        prompt = generate_correction_prompt(original_features, full_text)
        corrected = call_ollama(prompt)
        
        if corrected:
            print(f"Got corrections: {list(corrected.keys())}")
            corrected_row = row.copy()
            # For key features, use corrected if available, else original
            for key in key_features:
                corr_key = f"corrected_{key}"
                if key in corrected:
                    corrected_row[corr_key] = corrected[key]
                else:
                    corrected_row[corr_key] = row.get(key, np.nan)  # Keep original or NaN
            # Add notes
            corrected_row["correction_notes"] = corrected.get("correction_notes", "No changes made")
            corrected_batch.append(corrected_row)
        else:
            print(f"  Skipping {txt_filename}: No model response")
            # Add original row
            corrected_row = row.copy()
            corrected_row["correction_notes"] = "Skipped: Model failed"
            corrected_batch.append(corrected_row)
        
        time.sleep(SLEEP_TIME)
    
    return pd.DataFrame(corrected_batch)

# Define key_features globally for use in correct_features_batch
key_features = [
    "primary_nanomaterial", "primary_nano_type", "particle_size_nm", "zeta_potential_mV",
    "pzc_value", "primary_tumor_type", "primary_injection_method", "injection_dose_mg_kg",
    "injection_frequency", "ultrasound_frequency_MHz", "ultrasound_power_W_cm2",
    "treatment_time_min", "treatment_sessions", "ros_generation", "cell_viability_percent",
    "cell_death_percent", "tumor_volume_mm3"
]

# ========= Main Function =========
if __name__ == "__main__":
    # Load original features CSV
    if not os.path.exists(FEATURES_CSV):
        print(f"Original CSV does not exist: {FEATURES_CSV}")
        exit(1)
    
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} papers' original features")
    
    # Check if TEXT_DIR exists
    if not os.path.exists(TEXT_DIR):
        print(f"TEXT_DIR does not exist: {TEXT_DIR}")
        exit(1)
    
    # Batch processing
    corrected_rows = []
    total_batches = (len(df) - 1) // BATCH_SIZE + 1
    for i in range(0, len(df), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = df.iloc[i:i + BATCH_SIZE]
        print(f"\nProcessing batch {batch_num}/{total_batches}")
        corrected_batch = correct_features_batch(batch)
        corrected_rows.extend(corrected_batch.to_dict('records'))  # Use to_dict for extend
        print(f"Batch complete, cumulative {len(corrected_rows)} rows")
    
    # Build final DF
    df_corrected = pd.DataFrame(corrected_rows)
    
    # Save new CSV (includes original + corrected columns)
    df_corrected.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCorrection complete! New table saved to: {OUTPUT_CSV}")
    print(f"Contains {len(df_corrected)} papers' corrected features")
    
    # Simple stats: Compare before/after (handle NaN)
    if 'particle_size_nm' in df.columns and 'corrected_particle_size_nm' in df_corrected.columns:
        orig_mean = df['particle_size_nm'].dropna().mean()
        corr_mean = df_corrected['corrected_particle_size_nm'].dropna().mean()
        print(f"Example: Particle size mean - Original: {orig_mean:.2f} nm, Corrected: {corr_mean:.2f} nm")
    
    print("Now manually review the 'corrected_*' columns and 'correction_notes' in OUTPUT_CSV.")