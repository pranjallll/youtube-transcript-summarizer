import os
os.system("pip install streamlit==1.34.0")

import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai
from urllib.parse import urlparse, parse_qs
from yt_dlp import YoutubeDL
import requests
import re
import whisper
import tempfile
import glob

# ------------------------
# Load environment variables
# ------------------------
api_key = None

# 1️⃣ Try Streamlit secrets first
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

# 2️⃣ Fallback to local .env
if not api_key:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("❌ GOOGLE_API_KEY not found. Set it in .env (local) or Streamlit secrets (deployed)")

genai.configure(api_key=api_key)
print("Configured successfully ✅")

# ------------------------
# Helper to extract video ID
# ------------------------
def get_video_id(youtube_url: str) -> str:
    parsed = urlparse(youtube_url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/embed/") or parsed.path.startswith("/v/"):
            return parsed.path.split("/")[2]
    return None

# ------------------------
# Clean VTT subtitle text
# ------------------------
def vtt_to_text(vtt_data: str) -> str:
    lines = []
    for line in vtt_data.splitlines():
        if line.strip() and not line.startswith(("WEBVTT", "NOTE", "STYLE")) and not re.match(r"^\d+$", line):
            cleaned = re.sub(r"<.*?>", "", line.strip())  # remove HTML tags
            lines.append(cleaned)
    return " ".join(lines)

# ------------------------
# Cached Whisper model
# ------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

# ------------------------
# Extract transcript (captions OR Whisper)
# ------------------------
def extract_transcript(youtube_url: str, mode="captions", video_id=None):
    try:
        # 1) Try captions
        if mode == "captions":
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "subtitleslangs": ["en"],
                "subtitlesformat": "vtt",
                "quiet": True,
                "cachedir": False,   # 👈 don’t reuse bad state
                "http_headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/115.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            }

            # Load cookies.txt if exists
            if os.path.exists("cookies.txt"):
                ydl_opts["cookiefile"] = "cookies.txt"

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                subs = info.get("subtitles") or info.get("automatic_captions")
                if subs and "en" in subs:
                    sub_url = subs["en"][0]["url"]
                    headers = {
                        "User-Agent": ydl_opts["http_headers"]["User-Agent"],
                        "Accept-Language": ydl_opts["http_headers"]["Accept-Language"],
                    }
                    r = requests.get(sub_url, headers=headers, timeout=20)
                    r.raise_for_status()
                    transcript_text = vtt_to_text(r.text)
                    if transcript_text.strip():
                        return transcript_text

        # 2) Whisper fallback
        st.info("⚠️ No subtitles found, transcribing audio with Whisper... (this may take a while)")
        with tempfile.TemporaryDirectory() as tmpdir:
            outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")
            ydl_opts_audio = {
                "format": "140",  # m4a audio stream, usually safest
                "outtmpl": outtmpl,
                "noplaylist": True,
                "quiet": True,
                "cachedir": False,   # 👈 fresh every time
                "http_headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/115.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }],
            }

            if os.path.exists("cookies.txt"):
                ydl_opts_audio["cookiefile"] = "cookies.txt"

            with YoutubeDL(ydl_opts_audio) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                audio_path = glob.glob(os.path.join(tmpdir, "*.mp3"))[0]
                model = load_whisper_model()
                result = model.transcribe(audio_path)
                return result["text"]

    except Exception as e:
        st.error(f"⚠️ Failed to fetch transcript: {e}")
        return None

# ------------------------
# Gemini summary generator
# ------------------------
def generate_gemini_summary(transcript_text: str, video_id: str = None):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        uniqueness_hint = transcript_text[:300]
        prompt = f"""
        You are a YouTube video summarizer. 
        Summarize the following transcript in under 500 words.
        Keep it clear, concise, and highlight only the most important points.

        Video ID: {video_id or "unknown"}
        Uniqueness hint: {uniqueness_hint}

        Transcript:
        {transcript_text}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"⚠️ Failed to generate summary: {e}")
        return None

# ------------------------
# Streamlit UI
# ------------------------
st.title("🎥 YouTube Transcript to Summary Converter")

youtube_link = st.text_input("Enter YouTube Video Link:")
mode = st.radio("Select transcript method:", ["captions (fast, may fail)", "whisper (slow, reliable)"], index=0)

video_id = get_video_id(youtube_link) if youtube_link else None
if video_id:
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

if st.button("Get Summary"):
    if not youtube_link.strip():
        st.warning("⚠️ Please enter a valid YouTube link.")
    else:
        selected_mode = "captions" if "captions" in mode else "whisper"
        transcript_text = extract_transcript(youtube_link, mode=selected_mode, video_id=video_id)
        if transcript_text:
            st.markdown(f"### 🔎 Transcript Preview for Video ID {video_id}")
            st.text_area("Transcript", transcript_text[:1000], height=300)
            summary = generate_gemini_summary(transcript_text, video_id)
            if summary:
                st.markdown("## 📝 Video Summary:")
                st.write(summary)

 
 
                     
            
     

          
                 
