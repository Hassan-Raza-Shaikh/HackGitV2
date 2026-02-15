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
import google.generativeai as genai
import requests
import tempfile
from moviepy import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import os

# Use unverified SSL context so Whisper model download works behind corporate proxy / self-signed certs
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="EquiLearn", page_icon="🎓", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    /* Global Font & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #333;
    }
    
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
    }

    /* Video Player Container */
    .stVideo {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(100, 100, 250, 0.2);
        border: 4px solid #fff;
    }

    /* Active Segment Highlight */
    .active-segment {
        background: linear-gradient(90deg, #fff 0%, #f3e5f5 100%);
        border-left: 6px solid #d500f9;
        padding: 16px;
        margin-bottom: 12px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(213, 0, 249, 0.15);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        transform: scale(1.02);
    }

    /* Inactive Segment */
    .inactive-segment {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        padding: 14px;
        margin-bottom: 10px;
        border-radius: 12px;
        color: #666;
        transition: all 0.2s ease;
    }
    .inactive-segment:hover {
        border-color: #d500f9;
        transform: translateX(4px);
    }

    /* Simplify Card */
    .simplify-card {
        background: white;
        border-radius: 24px;
        padding: 24px;
        margin-top: 24px;
        border: 2px solid #e0e0e0;
        box-shadow: 8px 8px 0px rgba(0,0,0,0.05); /* Pop art style shadow */
        background-image: radial-gradient(#f3e5f5 1px, transparent 1px);
        background-size: 20px 20px;
    }

    .simplify-header {
        font-size: 1.4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #d500f9, #651fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    .keyword-tag {
        display: inline-block;
        background: linear-gradient(45deg, #ff4081, #d500f9);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
        margin-right: 8px;
        margin-bottom: 8px;
        box-shadow: 0 4px 10px rgba(213, 0, 249, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #651fff, #d500f9);
        color: white;
        border-radius: 12px;
        font-weight: 700;
        border: none;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 14px rgba(101, 31, 255, 0.4);
        transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .stButton > button:hover {
        transform: scale(1.05) translateY(-2px);
        box-shadow: 0 6px 20px rgba(101, 31, 255, 0.5);
        color: white;
    }
    .stButton > button:active {
        transform: scale(0.95);
    }

    /* Headers */
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(0deg, #2979ff, #d500f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3 {
        color: #37474f;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

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


def get_sign_url(word: str, signs_db: dict) -> str | None:
    """
    Find a sign URL for a word.
    1. Check local signs.json
    2. Try varying suffixes (e.g. remove 's', 'ing') if not found locally.
    3. Try fetching from Lifeprint (asl101.com) dynamically.
    """
    word = word.strip().lower()
    if not word:
        return None
        
    # 1. Local Lookup
    if word in signs_db:
        return signs_db[word]
    
    # Check session cache for dynamic lookups
    if "sign_url_cache" not in st.session_state:
        st.session_state.sign_url_cache = {}
    
    if word in st.session_state.sign_url_cache:
        return st.session_state.sign_url_cache[word]

    # 2. Dynamic Lookup (Lifeprint)
    # Common pattern: https://www.lifeprint.com/asl101/gifs/{first_letter}/{word}.gif
    first_letter = word[0]
    # Try exact word
    candidate_url = f"https://www.lifeprint.com/asl101/gifs/{first_letter}/{word}.gif"
    
    try:
        # Use a timeout to avoid hanging
        resp = requests.head(candidate_url, timeout=1.5)
        if resp.status_code == 200:
            st.session_state.sign_url_cache[word] = candidate_url
            return candidate_url
    except Exception:
        pass
        
    # Cache failure to avoid repeated bad requests
    st.session_state.sign_url_cache[word] = None
    return None


def match_keywords_to_gifs(keywords, signs):
    """
    Match extracted keywords to signs using dynamic lookup.
    Returns list of (keyword, gif_path_or_url) for each match.
    """
    if not keywords:
        return []
        
    matched = []
    seen = set()
    
    # Pre-load cache for batch
    if "sign_url_cache" not in st.session_state:
        st.session_state.sign_url_cache = {}

    for kw in keywords:
        key = kw.strip().lower()
        if not key or key in seen:
            continue
            
        url = get_sign_url(key, signs)
        if url:
            matched.append((kw.strip(), url))
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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except (FileNotFoundError, KeyError):
            pass

    if not api_key:
        st.sidebar.warning("⚠️ Enter Gemini API Key to enable AI features.")
    else:
        genai.configure(api_key=api_key)
        
        # --- Ask the Video (Chat) ---
        with st.sidebar:
            st.divider()
            st.header("💬 Ask the Video")
            
            # Display chat messages
            chat_container = st.container(height=300)
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
            
            # Chat Input
            if prompt := st.chat_input("Ask about the video..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.write(prompt)
                
                # Generate AI response
                if not st.session_state.segments:
                    response_text = "Please transcribe a video first so I can answer questions about it!"
                else:
                    # Construct context from transcript
                    full_transcript = " ".join([seg["text"] for seg in st.session_state.segments])
                    
                    try:
                        model = genai.GenerativeModel('gemini-flash-latest')
                        chat_prompt = (
                            f"System: You are an expert tutor. Answer the user's question based strictly on the following video transcript. "
                            f"If the answer is not in the transcript, say you don't know.\n\n"
                            f"Transcript: {full_transcript[:100000]}...\n\n" # Limit context to avoid token limits if extremely long
                            f"User Question: {prompt}"
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to .streamlit/secrets.toml or as an environment variable."
        )
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    
    # Updated Prompt for ASL Gloss
    # We ask for 'asl_gloss' which is a list of signs in order.
    # We also keep 'keywords' for backward compatibility or extra tagging if needed, 
    # but the primary focus for video is gloss.
    
    if accessibility_mode:
        prompt = (
            "System: Return only valid JSON: {\"explanation\": \"...\", \"asl_gloss\": [\"WORD1\", \"WORD2\", ...]}.\n"
            "User: Simplify in plain language. Short sentences. one example. "
            "Translate the meaning into ASL GLOSS (root words, ALL CAPS, time-topic-comment structure). "
            "Example: 'I went to store' -> ['ME', 'GO', 'STORE', 'FINISH'].\n"
            "JSON keys: explanation, asl_gloss.\n\n" + text.strip()
        )
    else:
        prompt = (
            "System: Return only valid JSON.\n"
            "User: Simplify text for general audience.\n"
            "- Short sentences, no jargon.\n"
            "- One concrete example.\n"
            "- Provide an ASL GLOSS translation of the summary (root words, ALL CAPS, grammar of American Sign Language).\n"
            "Return JSON with keys:\n"
            "  - \"explanation\": string\n"
            "  - \"asl_gloss\": list of strings (the sign sequence)\n\n"
            "Text to simplify:\n" + text.strip()
        )

    model = genai.GenerativeModel('gemini-flash-latest')
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        raw = response.text.strip()
        # Clean up potential markdown code blocks
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw)
    except Exception as e:
        return {
            "explanation": f"Error simplifying text: {str(e)}", 
            "keywords": [], 
            "asl_gloss": []
        }


# Session state
if "video_bytes" not in st.session_state:
    st.session_state.video_bytes = None
# Session state initialization
if "segments" not in st.session_state:
    st.session_state.segments = []
if "messages" not in st.session_state:
    st.session_state.messages = []
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
st.session_state.accessibility_mode = accessibility_mode # Store in session state for simplify_segment call

api_key_configured = os.environ.get("GEMINI_API_KEY") is not None
if not api_key_configured:
    try:
        if st.secrets["GEMINI_API_KEY"]:
            api_key_configured = True
    except (FileNotFoundError, KeyError):
        pass

if not api_key_configured:
    st.sidebar.caption("⚠️ Set **GEMINI_API_KEY** in `.streamlit/secrets.toml` or environment to enable features.")

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
    col_main, col_transcript = st.columns([1.8, 1.2], gap="large")

    with col_main:
        # 1. Video Player
        if video_bytes_for_player is not None:
            st.video(video_bytes_for_player)
        else:
            eid = st.session_state.yt_embed_id
            st.markdown(
                f'<iframe width="100%" height="400" src="https://www.youtube.com/embed/{eid}" '
                'style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" '
                'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
                unsafe_allow_html=True,
            )
        
        # 2. Seek Slider
        st.session_state.current_time = st.slider(
            "Seek (s)",
            0.0,
            max(st.session_state.duration, 1.0),
            st.session_state.current_time,
            0.5,
            label_visibility="collapsed"
        )

        # 3. Determine Active Segment
        active_idx = None
        for i, seg in enumerate(st.session_state.segments):
            if seg["start"] <= st.session_state.current_time < seg["end"]:
                active_idx = i
                break

        # 4. Simplify Section (Below Video)
        if active_idx is not None:
            seg = st.session_state.segments[active_idx]
            segment_text = seg["text"] # Renamed from 'text' to avoid conflict with prompt 'text'
            cache_key = f"{active_idx}:{hash(segment_text)}"
            
            st.markdown('<div class="simplify-card">', unsafe_allow_html=True)
            st.markdown('<div class="simplify-header">✨ AI Simplifier & Sign Language</div>', unsafe_allow_html=True)
            
            # Logic to fetch simplification
            should_simplify = False
            if cache_key in st.session_state.simplified_cache:
                should_simplify = True  # Already cached
            elif accessibility_mode:
                should_simplify = True  # Auto-simplify
            else:
                if st.button("Simplify Current Segment", key=f"simplify_btn_{active_idx}", use_container_width=True, type="primary"):
                    should_simplify = True

            if should_simplify and cache_key not in st.session_state.simplified_cache:
                with st.spinner("Simplifying..."):
                    try:
                        st.session_state.simplified_cache[cache_key] = simplify_segment(segment_text, accessibility_mode)
                    except Exception as e:
                        st.error(f"API Error: {e}")
                        st.session_state.simplified_cache[cache_key] = None

            # Display Simplification Results
            if cache_key in st.session_state.simplified_cache and st.session_state.simplified_cache[cache_key]:
                result = st.session_state.simplified_cache[cache_key]
                
                explanation = result.get("explanation", "Could not generate an explanation.")
                # Fallback: look for 'asl_gloss' first, then 'keywords'
                gloss_terms = result.get("asl_gloss", [])
                if not gloss_terms:
                    gloss_terms = result.get("keywords", [])
                
                st.markdown(f"<div class='simplify-header'>Explanation:</div> {explanation}", unsafe_allow_html=True)
                
                # Match GLOSS terms to Signs
                signs = load_signs() # Load signs here
                matched_gloss = match_keywords_to_gifs(gloss_terms, signs)
                
                if matched_gloss:
                    st.markdown("**ASL Gloss (Sign Sequence):**")
                    # Display tags for the gloss
                    tags_html = "".join([f"<span class='keyword-tag'>{term}</span>" for term, _ in matched_gloss])
                    st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
                    
                    # Generate Video Button
                    if st.button("🎥 Generate Sign Language Video"):
                        with st.spinner("Stitching sign language video..."):
                            video_path = generate_sign_video(matched_gloss)
                            if video_path:
                                st.video(video_path)
                                with open(video_path, "rb") as vf:
                                    st.download_button("Download Video", vf, "sign_language.mp4", "video/mp4")
                            else:
                                st.warning("Could not generate video. Sign GIFs might be unavailable.")

                    # Individual GIFs fallback (optional or distinct section)
                    with st.expander("View Individual Signs"):
                        cols = st.columns(4)
                        for i, (keyword, gif_path_or_url) in enumerate(matched_gloss):
                            with cols[i % 4]:
                                if gif_path_or_url.startswith("http"):
                                    st.image(gif_path_or_url, caption=keyword, use_container_width=True)
                                else:
                                    # Fallback for local files if any
                                    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), gif_path_or_url)
                                    if os.path.isfile(full_path):
                                        st.image(full_path, caption=keyword, use_container_width=True)
                else:
                    st.caption("No matching sign language GIFs found for this explanation.")
            elif not should_simplify and not accessibility_mode:
                st.info("Click 'Simplify' to get an AI explanation and ASL signs for this segment.")
                
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Play the video to see content here.")

    with col_transcript:
        st.subheader("Transcript")
        # Scrollable container for transcript
        with st.container(height=600):
            for i, seg in enumerate(st.session_state.segments):
                start, end, text = seg["start"], seg["end"], seg["text"]
                if not text:
                    continue
                
                is_active = (i == active_idx)
                # Scroll to active? Streamlit doesn't support programmatic scroll yet easily, 
                # but we can highlight visually.
                
                css_class = "active-segment" if is_active else "inactive-segment"
                timestamp = f"{int(start // 60)}:{int(start % 60):02d}"
                
                st.markdown(
                    f'''
                    <div class="{css_class}" id="seg-{i}">
                        <div style="font-size: 0.8rem; color: #888; margin-bottom: 4px;">{timestamp}</div>
                        <div style="font-size: 0.95rem; line-height: 1.4;">{text}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

else:
    st.info("Upload an MP4 file or paste a YouTube link to get started.")
    st.session_state.video_bytes = None
    st.session_state.segments = []
    st.session_state.current_time = 0.0
