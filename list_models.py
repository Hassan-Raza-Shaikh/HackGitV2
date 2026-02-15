import google.generativeai as genai
import os

# Load API key from secrets (simulating how app.py does it, but we can just ask user to set env for this script or read file)
import toml

try:
    with open(".streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)
        api_key = secrets.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            print("API Key configured.")
            
            print("Listing models...")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(m.name)
        else:
            print("No API key found in .streamlit/secrets.toml")
except Exception as e:
    print(f"Error: {e}")
