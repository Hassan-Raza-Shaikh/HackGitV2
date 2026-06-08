# HackGitV2 - Gemini API Sign Language Interpreter

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini_API-8E75C2?style=for-the-badge&logo=google&logoColor=white)

A Python back-end web application that leverages Google's Gemini API multimodal capabilities to analyze sign language visuals and translate them into text/speech.

## Key Features
*   **Flask Web Server**: Simple REST API hosting the translation endpoint.
*   **Gemini API Integration**: Multimodal prompt pipelines analyzing images and streams.
*   **Structured Metadata Mapping**: Predefined mappings for sign classifications in `signs.json`.

## Setup Instructions
1. Clone and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your API key:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
3. Run the Flask server:
   ```bash
   python app.py
   ```
