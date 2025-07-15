import os
import json
import subprocess
import pdfplumber
from openai import OpenAI
from pathlib import Path
from datetime import datetime

import warnings, logging
# Suppress pdfminer logging warnings
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FOLDER = "/mnt/d/Data/tdnet_pdfs/2025-06-23"
JSON_FOLDER = "test_output_evaluation"
REPORT_FOLDER = "evaluation_reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    try:
        # Use pdftotext for layout-preserving extraction
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Fallback to pdfplumber if pdftotext fails
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
        except Exception as fallback_e:
            return f"Error extracting text with both methods: pdftotext: {e}, pdfplumber: {fallback_e}"
    except Exception as e:
        return f"Error extracting text: {e}"

def load_json_text(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        
        if isinstance(content, list):
            # Extract content from chunks - this is the expected format
            extracted_chunks = []
            for chunk in content:
                if isinstance(chunk, dict) and "content" in chunk:
                    extracted_chunks.append(chunk["content"])
            return "\n\n".join(extracted_chunks) if extracted_chunks else "No content found in chunks"
        elif isinstance(content, dict):
            # Fallback for other formats
            return content.get("content", "") or content.get("markdown", "") or json.dumps(content)
        else:
            return str(content)
    except Exception as e:
        return f"Error loading JSON: {e}"

def call_llm_compare(original_text, chunks_json, filename, model="o3"):
    prompt = f"""You are "Extractor QA-bot", an expert that validates PDF-to-JSON
extraction for a financial-intelligence pipeline.

### Goal
Make sure JSON chunks are *optimal* for downstream LLM retrieval and
numeric parsing.  Detect anything that will break RAG similarity,
numerical rules, or provenance tracking.

### Inputs
------------------  <PDF_TEXT>  ------------------
{original_text[:30000]}
------------------  <END PDF_TEXT> ----------------

------------------  <CHUNKS_JSON>  ----------------
{chunks_json}
------------------  <END CHUNKS_JSON>  ------------

------------- <SCHEMA>  ----------------
disclosure_id, chunk_index, page_number, content_type,
content, metadata (with chunk_length, language, pdf_path, etc.)
-------------  <END SCHEMA>  ----------------------

### Tasks
1. **Exact duplicates**  
   *List any chunk indices whose `content` is byte-identical.*

2. **Near duplicates**  
   *Flag pairs whose content is >95 % similar or consists only of
   repeating page headers/footers.*

3. **Noise & artefacts**  
   *Detect and quote:*  
     • Repeating page banners (`ウェルネス・コミュニケーションズ`, "© 2025")  
     • Leader dots (`......`, `･････`) or table borders (`|`).  
     • Random OCR junk (isolated Latin runs, garbled numbers).  
     • Non-ASCII minus or percent signs (△ ▲ － − ％ full-width digits).

4. **Metadata sanity**  
   *Check every object for:*  
     • required keys present & non-null.  
     • `chunk_length` matches actual char count (if provided).  
     • `page_number` monotonic vs. PDF text shown.  
     • `content_type` variety (balance_sheet, cash_flow, etc.).

5. **Chunk-size distribution**  
   Report min/median/95th percentile length; list any chunk >1100
   JP characters or <120 chars.

6. **Actionable recommendations**  
   For every issue above, give **specific, code-implementable fixes** in
   bullets (e.g. regex or preprocessing step).

### Output format
Respond with **exactly** two top-level keys:

```json
{{
  "summary": "plain-English paragraph (<=150 words)",
  "issues": [
    {{
      "category": "duplicates",        // one of: duplicates, near_duplicates,
                                       // artefacts, metadata, sizing, other
      "detail": "short description",
      "example_chunks": [3, 5, 8],     // up to 5 indices
      "suggested_fix": "regex or algorithm"
    }}
  ]
}}
```"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # temperature=0,
            max_completion_tokens=8192
        )
        
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            if content:
                try:
                    # Extract JSON from markdown code blocks if present
                    json_content = content.strip()
                    if json_content.startswith('```json'):
                        # Remove markdown code blocks
                        lines = json_content.split('\n')
                        # Find start and end of JSON block
                        start_idx = 0
                        end_idx = len(lines)
                        for i, line in enumerate(lines):
                            if line.strip() == '```json':
                                start_idx = i + 1
                            elif line.strip() == '```' and i > start_idx:
                                end_idx = i
                                break
                        json_content = '\n'.join(lines[start_idx:end_idx])
                    
                    # Parse JSON response
                    return json.loads(json_content)
                except json.JSONDecodeError as e:
                    return {
                        "summary": f"Failed to parse LLM response as JSON: {e}",
                        "issues": [{"category": "other", "detail": f"Raw response: {content[:500]}...", "example_chunks": [], "suggested_fix": "Check LLM output format"}]
                    }
            else:
                return {
                    "summary": "No content returned from LLM",
                    "issues": []
                }
        else:
            return {
                "summary": "No choices in response",
                "issues": []
            }
            
    except Exception as e:
        return {
            "summary": f"Error calling LLM: {e}",
            "issues": []
        }

def evaluate_all_documents():
    pdf_files = list(Path(INPUT_FOLDER).glob("*.pdf"))

    summary_rows = []

    for pdf_path in pdf_files:
        filename = pdf_path.stem
        json_path = Path(JSON_FOLDER) / f"{filename}_chunks.json"
        if not json_path.exists():
            print(f"Missing JSON for {filename}, skipping.")
            continue

        print(f"Evaluating {filename}...")

        original_text = extract_text_from_pdf(pdf_path)
        chunks_json = load_json_text(json_path)
        
        # Load raw JSON for full analysis
        with open(json_path, "r", encoding="utf-8") as f:
            raw_chunks_json = f.read()
        
        result = call_llm_compare(original_text, raw_chunks_json, filename)

        # Write JSON report
        report_path = Path(REPORT_FOLDER) / f"{filename}_eval.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Add to summary
        summary = result.get("summary", "No summary available")
        issue_count = len(result.get("issues", []))
        summary_rows.append((filename, f"{issue_count} issues found: {summary[:100]}..."))

    # Write summary
    summary_path = Path(REPORT_FOLDER) / "summary_table.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Filename | Issues Summary\n")
        f.write("--- | ---\n")
        for filename, issues_summary in summary_rows:
            f.write(f"{filename} | {issues_summary}\n")

if __name__ == "__main__":
    evaluate_all_documents()
