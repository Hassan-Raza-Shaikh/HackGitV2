# HackGitV2

## EquiLearn

Minimal Streamlit app: upload MP4 → transcribe with Whisper → watch with segmented transcript.

**Requirements:** Python deps + **ffmpeg** (for Whisper). On macOS: `brew install ffmpeg`.

`.streamlit/config.toml` disables the file watcher to avoid a Streamlit/PyTorch conflict (torch `__path__` inspection).

```bash
pip install -r requirements.txt
streamlit run app.py
```

- **Upload** a short MP4
- **Transcription** uses OpenAI Whisper (base model; first run downloads the model)
- **Video player** + **segmented transcript** with timestamps
- **Seek slider** to jump to a time; current segment is highlighted (session state for pause/rewind tracking)
- **Simplify segment**: for the active segment, call OpenAI to get a plain-language explanation and 3–5 keywords (set `OPENAI_API_KEY` in env)
- **Sign GIFs**: keywords are matched against `signs.json` (sign name → GIF path or URL); matched GIFs are shown below the simplified explanation. Add GIFs in a `signs/` folder or use URLs in `signs.json`.

