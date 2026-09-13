import os
import json
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from the .env file
load_dotenv()

client = OpenAI() # OpenAI client automatically looks for the OPENAI_API_KEY environment variable.

# 2. Define prompt 
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
    """Reads a PDF and returns all the text."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def analyze_paper(text):
    response = client.chat.completions.create(
        model="gpt-4.1-mini", # "gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5-pro", 
                                # "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"
        #reasoning_effort="medium",
        response_format={ "type": "json_object" }, 
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the paper text:\n\n{text}"}
        ]
    )
    return json.loads(response.choices[0].message.content)

# 3. Process the folder
pdf_folder = "papers" # Name of your folder containing PDFs
all_results = {}

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        print(f"Processing {filename}...")
        pdf_path = os.path.join(pdf_folder, filename)
        
        # Extract text and analyze
        paper_text = extract_text_from_pdf(pdf_path)
        extracted_data = analyze_paper(paper_text)
        
        # Save results under the filename
        all_results[filename] = extracted_data

# 4. Save everything to one master file
with open("electrolyte_data_openai.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("Extraction complete! Check electrolyte_data_openai.json")