from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

for m in client.models.list():
    if "gemini" in m.name:
        model_id = m.name.split("/")[-1]
        print(model_id)