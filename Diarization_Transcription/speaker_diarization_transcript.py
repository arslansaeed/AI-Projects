"""
Speaker Diarization and Transcription Script
Uses Whisper for transcription and Pyannote for speaker identification
"""

import os
import warnings
import torch
import torchaudio
import whisper
import pyannote.audio
import huggingface_hub
import numpy as np
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Suppress warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress FP16/pooling warnings

# =====================================
# CONFIGURATION
# =====================================

class Config:
    load_dotenv()
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if not HUGGINGFACE_TOKEN:
        raise ValueError("HUGGINGFACE_TOKEN not found.")
       
    AUDIO_FILE = os.path.join(os.path.dirname(__file__),  "audio_files", "testAudio.mp3")
    #AUDIO_FILE: str = os.getenv("AUDIO_FILE", r"C:\MyCode\wajiha\testAudio.mp3")
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")
    OUTPUT_FILE: str = os.getenv("OUTPUT_FILE", "transcript_with_speakers.txt")
    NUM_SPEAKERS: Optional[int] = int(os.getenv("NUM_SPEAKERS", "0")) or None  # 0 = auto-detect
    LANGUAGE: str = os.getenv("LANGUAGE", "auto")  # Or "en", etc.


# =====================================
# FUNCTIONS
# =====================================


def check_gpu() -> bool:
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("⚠️  GPU not available, using CPU")
        return False


def convert_to_wav(audio_file: str) -> str:
    """Convert audio to WAV format at 16kHz mono if needed"""
    wav_file = "temp_converted.wav"
    try:
        print(f"🔄 Loading audio file: {audio_file}")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        waveform, sr = torchaudio.load(audio_file)
        print(f"   Original: {waveform.shape}, {sr}Hz")
        
        # Resample to 16kHz if needed
        if sr != 16000:
            print(f"   Resampling from {sr}Hz to 16000Hz")
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        
        # Convert to mono if stereo/multi-channel
        if waveform.shape[0] > 1:
            print("   Converting to mono (averaging channels)")
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Normalize to [-1, 1]
        waveform = waveform / waveform.abs().max().clamp(min=1.0)
        
        # Save as WAV
        torchaudio.save(wav_file, waveform, 16000)
        print(f"✅ Converted audio saved as: {wav_file}")
        return wav_file
    except Exception as e:
        print(f"❌ Error converting audio: {e}")
        if os.path.exists(wav_file):
            os.remove(wav_file)
        raise


def load_models(hf_token: str):
    """Load Whisper and Pyannote models"""
    print("\n📥 Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Load Whisper
    print(f"   Loading Whisper ({Config.WHISPER_MODEL_SIZE})...")
    whisper_model = whisper.load_model(Config.WHISPER_MODEL_SIZE, device=device)
    print("   ✅ Whisper loaded")
    
    # Debug versions (comment out after fix)
    print(f"   Whisper version: {whisper.__version__}")
    print(f"   Pyannote version: {pyannote.audio.__version__}")
    print(f"   HuggingFace Hub version: {huggingface_hub.__version__}")
    
    # Load Pyannote
    print("   Loading Pyannote diarization pipeline...")
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        diarization_pipeline.to(device)
        print(f"   Pipeline loaded on {device}")
        print("   ✅ Pyannote loaded")
    except Exception as e:
        print(f"\n❌ Error loading Pyannote: {e}")
        print("\nTroubleshooting:")
        print("1. Accept agreements: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("2. And: https://huggingface.co/pyannote/segmentation-3.0")
        print("3. Valid HF token with read access")
        raise
    
    return whisper_model, diarization_pipeline


def perform_diarization(pipeline: Pipeline, wav_file: str, num_speakers: Optional[int] = None) -> List[Dict[str, float]]:
    """Identify who spoke when, with optional speaker count"""
    print("\n🔊 Running speaker diarization...")
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    diarization = pipeline(wav_file, **kwargs)
    
    segments = [{"speaker": spk, "start": turn.start, "end": turn.end}
                for turn, _, spk in diarization.itertracks(yield_label=True)]
    
    if not segments:
        raise ValueError("No speaker segments detected—check audio quality/length")
    
    unique_speakers = set(s["speaker"] for s in segments)
    print(f"✅ Diarization complete! Found {len(unique_speakers)} speaker(s): {sorted(unique_speakers)}")
    return segments


def perform_transcription(model: Any, wav_file: str) -> List[Dict[str, Any]]:
    """Transcribe using raw waveform (bypasses FFmpeg entirely with language='en')"""
    print("\n✍️  Transcribing with Whisper...")
    try:
        # Load with torchaudio
        waveform, sr = torchaudio.load(wav_file)
        if sr != 16000:
            raise ValueError(f"Expected 16kHz after conversion, got {sr}")
        
        # To 1D np.float32 array, normalized [-1,1] with headroom
        audio = waveform.mean(dim=0).numpy()  # Average channels
        audio = (audio / np.max(np.abs(audio)) * 0.9).astype(np.float32)  # Normalize + headroom
        
        # Debug (remove after success)
        print(f"   Debug: audio type={type(audio)}, shape={audio.shape}, dtype={audio.dtype}, range=[{audio.min():.3f}, {audio.max():.3f}]")
        
        # Transcribe with raw array + explicit English (avoids 'auto' detection issues)
        result = model.transcribe(
            audio=audio,
            verbose=False,
            fp16=torch.cuda.is_available(),
            language="en",  # Fixed: Specify 'en' to skip detection
            task="transcribe"
        )
        
        segments_text = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in result["segments"]
        ]
        
        if not segments_text:
            raise ValueError("No transcription segments—check audio content")
        
        print(f"✅ Transcription complete! {len(segments_text)} segments (raw array mode)")
        return segments_text
    except Exception as e:
        print(f"❌ Raw array error: {e}")
        # Fallback only if needed (requires FFmpeg for file loading)
        print("   Falling back to file-based transcription...")
        try:
            result = model.transcribe(
                wav_file, 
                verbose=False, 
                fp16=torch.cuda.is_available(), 
                language="en",  # Fixed: Same for fallback
                task="transcribe"
            )
            segments_text = [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
                for seg in result["segments"]
            ]
            print(f"✅ Fallback success! {len(segments_text)} segments")
            return segments_text
        except Exception as fallback_e:
            print(f"❌ Fallback failed: {fallback_e}")
            print("💡 Install FFmpeg for file fallback: Download from https://www.gyan.dev/ffmpeg/builds/, extract to C:\\ffmpeg, add C:\\ffmpeg\\bin to PATH, restart terminal.")
            raise


def match_speaker(time: float, speaker_segments: List[Dict[str, float]], threshold: float = 1.0) -> str:
    """
    Fuzzy match timestamp to closest speaker.
    - Exact overlap preferred.
    - If within threshold (s), assign closest speaker.
    - Else, assign sequential "UnknownN".
    """
    global unknown_counter
    
    min_dist = float('inf')
    best_speaker = None
    
    for seg in speaker_segments:
        # Exact overlap
        if seg["start"] <= time <= seg["end"]:
            return seg["speaker"]
        
        # Distance to closest edge
        if time < seg["start"]:
            dist = seg["start"] - time
        else:
            dist = time - seg["end"]
        
        if dist < min_dist:
            min_dist = dist
            best_speaker = seg["speaker"]
    
    # If close enough, assign
    if min_dist <= threshold and best_speaker:
        return best_speaker
    
    # Sequential unknown
    speaker = f"Unknown{unknown_counter}"
    unknown_counter += 1
    print(f"   ⚠️  Unmatched time {time:.1f}s (dist={min_dist:.1f}s) → {speaker}")
    return speaker


def align_transcript_with_speakers(text_segments: List[Dict[str, Any]], speaker_segments: List[Dict[str, float]], threshold: float = 1.0) -> List[Dict[str, Any]]:
    """Align with fuzzy matching + sequential unknowns"""
    global unknown_counter
    unknown_counter = 1  # Reset per alignment
    
    print(f"\n🔗 Aligning text with speakers (threshold={threshold}s)...")
    
    final_transcript = []
    for seg in text_segments:
        if not seg["text"].strip():  # Skip empty
            continue
        
        mid_time = (seg["start"] + seg["end"]) / 2  # Or use seg["start"] for sequential bias
        speaker = match_speaker(mid_time, speaker_segments, threshold)
        
        final_transcript.append({
            "speaker": speaker,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })
    
    unique_speakers = set(t["speaker"] for t in final_transcript)
    unknown_count = sum(1 for s in unique_speakers if s.startswith("Unknown"))
    print(f"✅ Alignment complete! {len(final_transcript)} segments, {len(unique_speakers)} unique speakers ({unknown_count} unknowns)")
    if unknown_count > 0:
        print(f"   Speakers: {sorted(unique_speakers)}")
    
    return final_transcript


def save_transcript(transcript: List[Dict[str, Any]], output_file: str):
    """Save with timestamps"""
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Speaker Diarization & Transcription\n")
        f.write("=" * 50 + "\n\n")
        for item in transcript:
            timestamp = f"[{item['start']:.1f}s - {item['end']:.1f}s]"
            f.write(f"{item['speaker']} {timestamp}: {item['text']}\n")
    print("✅ Saved!")


def display_transcript(transcript: List[Dict[str, Any]]):
    """Console display with grouping"""
    print("\n" + "="*60)
    print("🗣️  FINAL TRANSCRIPT")
    print("="*60 + "\n")
    current_speaker = None
    for item in transcript:
        if item['speaker'] != current_speaker:
            current_speaker = item['speaker']
            print(f"\n{current_speaker.upper()}:")
        print(f"  [{item['start']:.1f}s] {item['text']}")
    print("\n" + "="*60)

# =====================================
# MAIN EXEC
# =====================================


def main():
    print("\n" + "="*60)
    print("🎙️  SPEAKER DIARIZATION & TRANSCRIPTION")
    print("="*60)
    
    check_gpu()
    
    wav_file = convert_to_wav(Config.AUDIO_FILE)
    try:
        whisper_model, diarization_pipeline = load_models(Config.HUGGINGFACE_TOKEN)
        
        speaker_segments = perform_diarization(diarization_pipeline, wav_file, Config.NUM_SPEAKERS)
        
        text_segments = perform_transcription(whisper_model, wav_file)
        
        final_transcript = align_transcript_with_speakers(text_segments, speaker_segments)
        
        display_transcript(final_transcript)
        save_transcript(final_transcript, Config.OUTPUT_FILE)
        
        print("\n✅ Process complete!")
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)
            print(f"🗑️  Cleaned up: {wav_file}")


if __name__ == "__main__":
    main()



