import google.generativeai as genai
import os
import toml

try:
    with open(".streamlit/secrets.toml", "r") as f:
        secrets = toml.load(f)
        api_key = secrets.get("GEMINI_API_KEY")

    if not api_key:
        print("No API key found.")
        exit(1)

    genai.configure(api_key=api_key)
    # Trying the model we set in app.py
    model = genai.GenerativeModel('gemini-flash-latest')
    
    print("Sending request to gemini-flash-latest...")
    response = model.generate_content("Hello, are you a free model?")
    print(f"Response: {response.text}")

except Exception as e:
    print(f"Error: {e}")
