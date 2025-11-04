
````markdown
# Diarization_Transcription



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

**Windows (PowerShell):**

```powershell
python -m venv venv
```

**Windows (cmd):**

```cmd
python -m venv venv
```

**macOS / Linux:**

```bash
python3 -m venv venv
```

This will create a folder named `venv` in your project directory.

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

Make sure your `requirements.txt` file is in the **root project directory** (same level as `venv`).

Install all required packages with:

```bash
pip install -r requirements.txt
```

---

### 5. Verify Installation

Check that all packages are installed:

```bash
pip list
```

You should see all packages listed in `requirements.txt`.

---

### 6. Run Your Code

After activating the virtual environment and installing dependencies:

```bash
python speaker_diarization_local.py
```

or open your notebook in VS Code/Jupyter and run the cells.

---
### 7. if you update `requirements.txt` you will need to rerun
pip install -r requirements.txt


---
### 8. Updating `requirements.txt` from installed environment.

If you install new packages in your venv, update `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

### ⚡ Notes

* Never place `requirements.txt` inside the `venv/` folder.
* Always activate the virtual environment before running your scripts.
* For Python 3.11 users: ensure PyTorch, TorchVision, and Torchaudio versions are compatible.

```


