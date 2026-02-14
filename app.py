"""
EquiLearn - Minimal hackathon app: upload MP4 or paste YouTube link → transcribe with Whisper → segmented transcript.
"""
import json
import os
import re
import shutil
import ssl
import tempfile
import warnings

# Suppress Whisper FP16-on-CPU warning (expected when no GPU)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU", module="whisper")

import streamlit as st
import whisper
from openai import OpenAI

# Use unverified SSL context so Whisper model download works behind corporate proxy / self-signed certs
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="EquiLearn", page_icon="🎓", layout="wide")

SIGNS_PATH = os.path.join(os.path.dirname(__file__), "signs.json")


def ffmpeg_available() -> bool:
    """Return True if ffmpeg is on PATH (required by Whisper to load audio)."""
    return shutil.which("ffmpeg") is not None


def youtube_video_id(url: str) -> str | None:
    """Extract YouTube video ID from URL."""
    if not url or not url.strip():
        return None
    patterns = [
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url.strip())
        if m:
            return m.group(1)
    return None


def download_youtube_audio(url: str) -> str | None:
    """Download YouTube video/audio to a temp file. Returns path or None on failure."""
    try:
        import yt_dlp
    except ImportError:
        return None
    vid = youtube_video_id(url)
    if not vid:
        return None
    out_dir = tempfile.mkdtemp()
    out_path = os.path.join(out_dir, f"{vid}.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    # find the downloaded file
    for f in os.listdir(out_dir):
        if f.startswith(vid):
            return os.path.join(out_dir, f)
    return None


def load_signs():
    """Load sign name -> GIF path/URL from signs.json. Returns dict (empty if missing)."""
    if not os.path.isfile(SIGNS_PATH):
        return {}
    with open(SIGNS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def match_keywords_to_gifs(keywords, signs):
    """
    Match extracted keywords to signs (case-insensitive).
    Returns list of (keyword, gif_path_or_url) for each match.
    """
    if not keywords or not signs:
        return []
    matched = []
    seen = set()
    for kw in keywords:
        key = kw.strip().lower()
        if not key or key in seen:
            continue
        if key in signs:
            matched.append((kw.strip(), signs[key]))
            seen.add(key)
    return matched


def simplify_segment(text: str, accessibility_mode: bool = False) -> dict:
    """
    Use OpenAI API to simplify segment text.
    Returns: {"explanation": str, "keywords": list[str]} (3–5 keywords).
    When accessibility_mode=True uses a shorter, simpler prompt.
    """
    if not text or not text.strip():
        return {"explanation": "", "keywords": []}
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is not set. Add your API key to the environment, e.g. "
            "export OPENAI_API_KEY=sk-... in the terminal before running Streamlit."
        )
    client = OpenAI()
    if accessibility_mode:
        system = "Return only valid JSON: {\"explanation\": \"...\", \"keywords\": [\"a\", \"b\", ...]}."
        user = (
            "Simplify in plain language. Short sentences. One example. "
            "3–5 keywords. JSON with keys explanation and keywords.\n\n" + text.strip()
        )
    else:
        system = "You return only valid JSON, no markdown or extra text."
        user = """Simplify this text for a general audience. Rules:
- Use short sentences only.
- Avoid jargon; use plain language.
- Include exactly one simple, concrete example.
- Return valid JSON with two keys only:
  - "explanation": your simplified explanation (one or two short paragraphs).
  - "keywords": a list of 3 to 5 key terms (strings).

Text to simplify:
""" + text.strip()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    out = json.loads(raw)
    return {
        "explanation": out.get("explanation", ""),
        "keywords": out.get("keywords", [])[:5],
    }


# Session state
if "video_bytes" not in st.session_state:
    st.session_state.video_bytes = None
if "segments" not in st.session_state:
    st.session_state.segments = []
if "current_time" not in st.session_state:
    st.session_state.current_time = 0.0
if "duration" not in st.session_state:
    st.session_state.duration = 0.0
if "simplified_cache" not in st.session_state:
    st.session_state.simplified_cache = {}
if "last_source_id" not in st.session_state:
    st.session_state.last_source_id = None
if "yt_embed_id" not in st.session_state:
    st.session_state.yt_embed_id = None

# Sidebar: Accessibility Mode
accessibility_mode = st.sidebar.toggle("Accessibility Mode", value=False, help="Auto-simplify segments, show sign GIFs, use simpler prompt.")
if not os.environ.get("OPENAI_API_KEY"):
    st.sidebar.caption("⚠️ Set **OPENAI_API_KEY** in your environment to enable **Simplify** and Accessibility Mode.")

st.title("🎓 EquiLearn")
st.caption("Upload an MP4 or paste a YouTube link → transcribe with Whisper → watch with segmented transcript.")

input_mode = st.radio("Input", ["Upload MP4", "Paste YouTube link"], horizontal=True, label_visibility="collapsed")

media_path = None
source_id = None
video_bytes_for_player = None
yt_embed_id = None

if input_mode == "Upload MP4":
    uploaded = st.file_uploader("Upload MP4 video", type=["mp4"], help="Short videos work best for quick transcription.")
    if uploaded:
        st.session_state.video_bytes = uploaded.read()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(st.session_state.video_bytes)
            media_path = tmp.name
        source_id = uploaded.file_id
        video_bytes_for_player = st.session_state.video_bytes
else:
    yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...")
    load_yt = st.button("Load & transcribe")
    if yt_url and load_yt:
        vid = youtube_video_id(yt_url)
        if not vid:
            st.error("Could not parse YouTube URL. Use a standard watch or youtu.be link.")
        else:
            with st.spinner("Downloading from YouTube…"):
                media_path = download_youtube_audio(yt_url)
            if not media_path:
                st.error("Download failed. Install yt-dlp: pip install yt-dlp")
            else:
                source_id = yt_url
                st.session_state.yt_embed_id = vid
    elif st.session_state.get("yt_embed_id") and st.session_state.segments:
        source_id = st.session_state.get("last_source_id")
        yt_embed_id = st.session_state.yt_embed_id

if media_path is not None and source_id is not None:
    if not st.session_state.segments or st.session_state.get("last_source_id") != source_id:
        if not ffmpeg_available():
            st.error(
                "**ffmpeg** is required for transcription but was not found. Install it, then restart the app.\n\n"
                "- **macOS:** `brew install ffmpeg`\n"
                "- **Windows:** [Download from ffmpeg.org](https://ffmpeg.org/download.html) or `winget install ffmpeg`\n"
                "- **Linux:** `sudo apt install ffmpeg` or `sudo dnf install ffmpeg`"
            )
        else:
            with st.spinner("Transcribing with Whisper (base model)…"):
                model = whisper.load_model("base")
                result = model.transcribe(media_path, word_timestamps=False)
            st.session_state.segments = [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result["segments"]
            ]
            st.session_state.duration = max((s["end"] for s in st.session_state.segments), default=0.0)
            st.session_state.last_source_id = source_id
            st.session_state.current_time = 0.0
            st.session_state.simplified_cache = {}
            if yt_embed_id:
                st.session_state.yt_embed_id = yt_embed_id

if (video_bytes_for_player is not None or st.session_state.get("yt_embed_id")) and st.session_state.segments:
    col_video, col_transcript = st.columns([1, 1])

    with col_video:
        if video_bytes_for_player is not None:
            st.video(video_bytes_for_player)
        else:
            eid = st.session_state.yt_embed_id
            st.markdown(
                f'<iframe width="100%" height="315" src="https://www.youtube.com/embed/{eid}" '
                'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
                unsafe_allow_html=True,
            )
        st.session_state.current_time = st.slider(
            "Seek (s)",
            0.0,
            max(st.session_state.duration, 1.0),
            st.session_state.current_time,
            0.5,
            help="Move to sync transcript with playback.",
        )

    with col_transcript:
        st.subheader("Transcript (segments)")
        active_idx = None
        for i, seg in enumerate(st.session_state.segments):
            start, end, text = seg["start"], seg["end"], seg["text"]
            if not text:
                continue
            is_active = start <= st.session_state.current_time < end
            if is_active:
                active_idx = i
            st.markdown(
                f'<div style="padding: 0.5rem 0.75rem; margin: 0.25rem 0; border-radius: 6px; '
                f'background: {"#e3f2fd" if is_active else "#f5f5f5"}; '
                f'border-left: 4px solid {"#1976d2" if is_active else "transparent"}; '
                f'font-size: 0.95rem;">'
                f'<span style="color: #666; font-size: 0.8rem;">[{start:.1f}s – {end:.1f}s]</span> {text}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Simplified explanation + keywords for active segment
        if active_idx is not None:
            seg = st.session_state.segments[active_idx]
            text = seg["text"]
            cache_key = f"{active_idx}:{hash(text)}"
            if cache_key not in st.session_state.simplified_cache:
                if accessibility_mode:
                    with st.spinner("Simplifying…"):
                        try:
                            st.session_state.simplified_cache[cache_key] = simplify_segment(text, accessibility_mode=True)
                        except Exception as e:
                            st.error(f"API error: {e}")
                            st.session_state.simplified_cache[cache_key] = None
                elif st.button("Simplify this segment", key="simplify_btn"):
                    with st.spinner("Simplifying…"):
                        try:
                            st.session_state.simplified_cache[cache_key] = simplify_segment(text)
                        except Exception as e:
                            st.error(f"API error: {e}")
                            st.session_state.simplified_cache[cache_key] = None
            if cache_key in st.session_state.simplified_cache and st.session_state.simplified_cache[cache_key]:
                data = st.session_state.simplified_cache[cache_key]
                st.markdown("---")
                st.markdown("**Simplified**")
                st.markdown(data["explanation"])
                if data.get("keywords"):
                    st.caption("Keywords: " + ", ".join(data["keywords"]))

                signs = load_signs()
                matched = match_keywords_to_gifs(data.get("keywords", []), signs)
                if matched:
                    st.markdown("**Signs**")
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    cols = st.columns(len(matched))
                    for i, (keyword, gif_path_or_url) in enumerate(matched):
                        with cols[i]:
                            if gif_path_or_url.startswith("http"):
                                st.image(gif_path_or_url, caption=keyword, use_container_width=True)
                            else:
                                full_path = os.path.join(base_dir, gif_path_or_url)
                                if os.path.isfile(full_path):
                                    st.image(full_path, caption=keyword, use_container_width=True)
                                else:
                                    st.caption(f"{keyword} (no GIF)")

else:
    st.info("Upload an MP4 file or paste a YouTube link to get started.")
    st.session_state.video_bytes = None
    st.session_state.segments = []
    st.session_state.current_time = 0.0
