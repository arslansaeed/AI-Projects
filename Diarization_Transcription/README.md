````markdown
# Diarization & Transcription

Turn raw audio into **speaker-labeled transcripts** offline, without cloud costs or API limits. This project loads MP3/WAV files, diarizes speakers using **Pyannote.audio**, transcribes with **Whisper**, aligns transcripts to speakers, and outputs clean TXT files. Perfect for meetings, interviews, or doctor appointments.

---

##  Features

- Offline speaker diarization and transcription  
- CPU-friendly, no cloud API required  
- Handles real-world audio with multiple speakers  
- Outputs neatly aligned speaker-labeled TXT files  
- Easy to run in a local Python environment  

---

## 🛠 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/arslansaeed/AI-Projects.git
cd Diarization_Transcription
````

---

### 2. Create a Virtual Environment (venv)

It's recommended to use a virtual environment to isolate dependencies.

**Windows (PowerShell or cmd):**

```powershell
py -3.11 -m venv venv
```

**macOS / Linux:**

```bash
python3.11 -m venv venv
```

This creates a folder named `venv` in your project directory.

---

### 3. Activate the Virtual Environment

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```cmd
.\venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

> You should see `(venv)` at the start of your terminal prompt once it’s active.

---

### 4. Install Dependencies

Ensure `requirements.txt` is in the **root project directory** (same level as `venv`).

```bash
pip install -r requirements.txt
```

---

### 5. Verify Installation

```bash
pip list
```

Make sure all packages from `requirements.txt` are installed.

---

### 6. Run the Project

```bash
python speaker_diarization_local.py
```

Or open the notebook in VS Code/Jupyter and run the cells interactively.

---

### 7. Updating Dependencies

If you modify `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you install new packages in your venv, update `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

###  Notes & Tips

* Never place `requirements.txt` inside the `venv/` folder.
* Always activate the virtual environment before running scripts.
* Ensure **PyTorch**, **TorchVision**, and **Torchaudio** versions are compatible with Python 3.11 (64-bit).
* Use `pip check` to verify dependency compatibility after installation.

---

## Challenges I Faced

I had both **Python 3.13** and **Python 3.11 (32-bit)** installed on my system. I didn’t realize my 3.11 version was **32-bit**, which caused hours of frustrating debugging. 

Upgrading Python often comes with compatibility headaches, especially with **GPU-heavy libraries** like PyTorch, Torchaudio, and Pyannote.audio. Wheels failed to install, dependencies clashed, and runtime errors kept popping up.

After switching to **Python 3.11 64-bit** and carefully realigning `requirements.txt`, everything finally ran smoothly—including PyTorch, Torchaudio, TorchVision, Pyannote, and NumPy.

**Key Takeaways:**

* Always verify Python architecture (64-bit vs 32-bit)
* Tools like pip debug --verbose and python -c "import sys; print(sys.maxsize > 2**32)" are lifesavers for wheel/tag mismatches.
* Pin library versions to avoid unexpected conflicts
* Sometimes the “latest version” isn’t worth the pain; stability saves time and sanity

---

## File Structure

* `speaker_diarization_local.py` – main script for diarization and transcription
* `requirements.txt` – Python dependencies

---

## Usage Example

```python
from speaker_diarization_local import diarize_and_transcribe

# Load your audio file
audio_file = "example_audio.mp3"

# Run diarization and transcription
diarize_and_transcribe(audio_file, output_file="output_transcript.txt")
```

> This will generate a speaker-labeled transcript offline, ready to use.

---


