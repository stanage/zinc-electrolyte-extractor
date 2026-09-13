import os
import json
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()

SYSTEM_PROMPT = """
System Role:
You are an expert materials science researcher specializing in molecular dynamics (MD) simulations and battery chemistries. Your task is to carefully analyze the provided text from a scientific paper and extract specific data regarding zinc battery electrolyte formulations.
Task 1: Electrolyte Data Extraction
Carefully scan the text for any detailed zinc battery electrolyte formulations. If the paper details the electrolyte used for experiments or simulations, extract the following variables. If multiple electrolytes are tested (e.g., a baseline and a novel formulation), extract this data for each one. If a specific variable is not mentioned, output "Not Specified".

Salt(s): The chemical name or formula of the zinc salt(s) (e.g., Zn(OTf)2, ZnSO4).
Solvent(s): The primary solvent(s) used (e.g., H2O, PC, DMC).
Additive(s): Any specific chemical additives introduced to the electrolyte.
Salt Concentration: The concentration of the salt(s), specifically in Molarity (M) or mol/kg (m) if M is unavailable.
Solvent Ratio: The volume ratio (v/v) or weight ratio (w/w) if multiple solvents are used.
Additive Amount: The amount of additive used, specifically in weight percent (wt%) of the electrolyte, or other explicitly stated units.
Task 2: Reference Mining
Scan the text for references to other previous studies specifically working on zinc batteries that are highly likely to contain their own detailed electrolyte formulations. Focus on sentences discussing previous electrolyte designs, solvation structures, zinc cell performance, or zinc reversibility. Extract these references.
Output Format:
Provide your response strictly in the following JSON structure to allow for automated parsing. Do not include introductory or concluding conversational text.

{
  "extracted_electrolytes": [
    {
      "electrolyte_designation": "Name or identifier (e.g., Baseline, Electrolyte A)",
      "salts": [],
      "solvents": [],
      "additives": [],
      "salt_concentration": "",
      "solvent_ratio": "",
      "additive_amount": ""
    }
  ],
  "potential_references": [
    {
      "in_text_citation": "e.g., [14] or (Smith et al., 2021)",
      "context_sentence": "Exact sentence from the text mentioning this reference.",
      "reason_for_inclusion": "Brief explanation of why this likely contains electrolyte MD/experimental data."
    }
  ]
}
"""

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a local PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def analyze_paper_with_gemini(paper_text):
    """Creates a chat session to process the paper and enforce valid JSON output."""
    chat = client.chats.create(
        model="gemini-3.6-flash", # gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash, 
                                    #gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-3.1-flash-lite-preview, 
                                    # gemini-3-flash-preview, gemini-flash-latest, gemini-flash-lite-latest, 
                                    # gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-pro-latest
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    
    prompt = f"{SYSTEM_PROMPT}\n\n--- PAPER TEXT ---\n{paper_text}"
    response = chat.send_message(prompt)
    
    return json.loads(response.text)

# 3. Process PDF folder
pdf_folder = "papers"
all_results = {}

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        print(f"Analyzing {filename} with Gemini...")
        pdf_path = os.path.join(pdf_folder, filename)
        
        try:
            text = extract_text_from_pdf(pdf_path)
            data = analyze_paper_with_gemini(text)
            all_results[filename] = data
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# 4. Save results
with open("electrolyte_data_gemini.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("Finished! Results saved to electrolyte_data_gemini.json")