import streamlit as st
from dotenv import load_dotenv
import os
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
load_dotenv()
genai.configure(api_key=os.getenv("Google_API_KEY"))

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
# Extract transcript (captions OR Whisper)
# ------------------------
def extract_transcript(youtube_url: str, mode="captions", video_id=None):
    try:
        # ---------- 1) Try captions ----------
        if mode == "captions":
            ydl_opts = {
                "skip_download": True,
                "writesubtitles": True,
                "subtitleslangs": ["en"],
                "subtitlesformat": "vtt",
                "quiet": True,
                "nocache": True,
                "cachedir": False,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                subs = info.get("subtitles") or info.get("automatic_captions")

                if subs and "en" in subs:
                    sub_url = subs["en"][0]["url"]
                    r = requests.get(sub_url)
                    r.raise_for_status()
                    transcript_text = "\n".join(
                        re.sub(r"<.*?>", "", line)
                        for line in r.text.splitlines()
                        if line.strip() and not line.startswith("WEBVTT") and not re.match(r"^\d+$", line)
                    )
                    if transcript_text.strip():
                        return transcript_text

        # ---------- 2) Whisper fallback ----------
        st.info("⚠️ No subtitles found, transcribing audio with Whisper... (this may take a while)")

        with tempfile.TemporaryDirectory() as tmpdir:
            outtmpl = os.path.join(tmpdir, "%(id)s.%(ext)s")

            ydl_opts_audio = {
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
                "noplaylist": True,
                "quiet": True,
                "nocache": True,
                "cachedir": False,
                "ffmpeg_location": "/opt/homebrew/bin",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
            }

            with YoutubeDL(ydl_opts_audio) as ydl:
                info = ydl.extract_info(youtube_url, download=True)

            vid = info.get("id") or (video_id or "unknown")
            final_audio = os.path.join(tmpdir, f"{vid}.mp3")

            if not os.path.exists(final_audio):
                mp3_candidates = glob.glob(os.path.join(tmpdir, "*.mp3"))
                if mp3_candidates:
                    final_audio = max(mp3_candidates, key=os.path.getmtime)
                else:
                    raise FileNotFoundError("No MP3 produced by yt-dlp in temp dir.")

            try:
                size = os.path.getsize(final_audio)
            except Exception:
                size = -1
            st.caption(f"🔧 Whisper debug → tmpdir: {tmpdir} | video_id: {vid} | audio: {os.path.basename(final_audio)} | size: {size} bytes")

            if size <= 0:
                raise RuntimeError("Downloaded audio file is empty.")

            model = whisper.load_model("base")
            result = model.transcribe(final_audio, fp16=False)
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

mode = st.radio(
    "Select transcript method:",
    ["captions (fast, may fail)", "whisper (slow, reliable)"],
    index=0
)

video_id = None
if youtube_link:
    video_id = get_video_id(youtube_link)
    if video_id:
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

if st.button("Get Summary"):
    if not youtube_link.strip():
        st.warning("⚠️ Please enter a valid YouTube link.")
    else:
        if "captions" in mode:
            selected_mode = "captions"
        elif "whisper" in mode:
            selected_mode = "whisper"
        else:
            selected_mode = "auto"

        transcript_text = extract_transcript(youtube_link, mode=selected_mode, video_id=video_id)
        if transcript_text:
            st.markdown(f"### 🔎 Debug: Transcript Preview for Video ID {video_id}")
            st.write(transcript_text[:500])

            summary = generate_gemini_summary(transcript_text, video_id)
            if summary:
                st.markdown("## 📝 Video Summary:")
                st.write(summary)
