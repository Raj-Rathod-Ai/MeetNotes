import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional
import static_ffmpeg
import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError
from pydub import AudioSegment

# Initialize static ffmpeg binaries
static_ffmpeg.add_paths()

FFMPEG_PATH = shutil.which("ffmpeg")
FFMPEG_DIR = os.path.dirname(FFMPEG_PATH) if FFMPEG_PATH else None

if FFMPEG_PATH:
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg = FFMPEG_PATH


def auto_purge_old_temp_files(max_age_seconds: int = 600) -> None:
    """
    Auto-cleans any temporary meeting files created more than 10 minutes ago.
    Prevents storage buildup on free hosting (Render).
    """
    temp_base = Path(tempfile.gettempdir())
    current_time = time.time()
    
    for pattern in ["meetnote_*", "vortex_*"]:
        for folder in temp_base.glob(pattern):
            try:
                if folder.is_dir() and (current_time - folder.stat().st_mtime > max_age_seconds):
                    shutil.rmtree(folder, ignore_errors=True)
            except Exception:
                pass


def download_youtube(url: str, output_dir: Optional[str] = None) -> str:
    """
    Download lowest bandwidth audio-only stream from YouTube into ephemeral storage.
    """
    auto_purge_old_temp_files()
    
    target_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="meetnote_yt_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    
    out_template = str(target_dir / "%(id)s.%(ext)s")
    
    options = {
        # Select best lightweight audio stream (keeps download sizes tiny < 5MB)
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": out_template,
        "ffmpeg_location": FFMPEG_DIR or FFMPEG_PATH,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "96",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get("id", "audio")
    except DownloadError as err:
        error_msg = str(err)
        if "Video unavailable" in error_msg:
            raise ValueError("This YouTube video is unavailable (it may be private, deleted, or region-blocked). Please check the link or upload the recording directly.")
        elif "Sign in" in error_msg or "age-restricted" in error_msg:
            raise ValueError("This YouTube video is age-restricted or requires login. Please upload the file directly.")
        else:
            raise ValueError(f"Could not access YouTube video: {error_msg}")
    except Exception as exc:
        raise ValueError(f"YouTube extraction error: {exc}")
        
    wav_file = target_dir / f"{video_id}.wav"
    if not wav_file.exists():
        candidates = list(target_dir.glob(f"*{video_id}*.wav"))
        if candidates:
            wav_file = candidates[0]
            
    return str(wav_file.resolve())


def standardize_audio(file_path: str, output_dir: Optional[str] = None) -> str:
    """Resample any media format to 16kHz mono WAV for Whisper."""
    path_obj = Path(file_path)
    target_dir = Path(output_dir) if output_dir else path_obj.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = target_dir / f"{path_obj.stem}_16k.wav"

    if str(path_obj.resolve()) == str(output_path.resolve()):
        return str(output_path)

    sound = AudioSegment.from_file(file_path)
    sound = sound.set_channels(1).set_frame_rate(16000)
    sound.export(str(output_path), format="wav")
    return str(output_path)


def split_audio_chunks(wav_path: str, chunk_minutes: int = 10) -> List[str]:
    """Segment audio tracks into 10-minute slices."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_len_ms = chunk_minutes * 60 * 1000

    if len(audio) <= chunk_len_ms:
        return [wav_path]

    stem = Path(wav_path).stem
    parent = Path(wav_path).parent
    chunk_files = []

    for idx, start_ms in enumerate(range(0, len(audio), chunk_len_ms)):
        segment = audio[start_ms : start_ms + chunk_len_ms]
        part_path = parent / f"{stem}_part_{idx}.wav"
        segment.export(str(part_path), format="wav")
        chunk_files.append(str(part_path))

    return chunk_files


def process_input(source: str, chunk_minutes: int = 10, work_dir: Optional[str] = None) -> List[str]:
    """Standardized entrypoint to prepare audio tracks from URL or local path."""
    source_clean = source.strip()
    if source_clean.startswith("http://") or source_clean.startswith("https://"):
        raw_audio = download_youtube(source_clean, output_dir=work_dir)
    else:
        raw_audio = source_clean

    mono_16k = standardize_audio(raw_audio, output_dir=work_dir)
    return split_audio_chunks(mono_16k, chunk_minutes=chunk_minutes)


def cleanup_files(file_paths: List[str]) -> None:
    """Immediately remove temporary audio files from disk to keep Render storage empty."""
    for path in file_paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass