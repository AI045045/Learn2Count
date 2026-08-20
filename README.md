# Learn2Count
# ✏️ Learn2Count — Handwritten Digit Predictor

A real-time handwritten digit recognition app using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. Draw a digit on the canvas and get instant predictions!

---

## 🧠 How It Works

1. Trains a CNN model on the MNIST handwritten digits dataset (10 epochs)
2. Saves the trained model as `handwritten_cnn.keras`
3. Opens an OpenCV drawing canvas
4. You draw a digit → right-click → the model predicts it in the terminal

> 💡 Training only happens on the **first run**. After that, the saved model is loaded instantly.

---

## 📋 Prerequisites

- Python **3.11** (via Homebrew recommended — avoids macOS LibreSSL conflicts)
- pip (Python package manager)
- Git (optional, for cloning)

---

## 🍎 Setup on macOS

> ⚠️ **Important**: Do NOT use the system Python (`/usr/bin/python3`) — it uses Apple's LibreSSL which conflicts with TensorFlow. Use Homebrew Python 3.11 instead.

### 1. Install Homebrew Python 3.11 (if not already installed)
```bash
brew install python@3.11
```

### 2. Clone or download the project
```bash
git clone <your-repo-url>
cd Learn2Count
```

### 3. Create a virtual environment with Python 3.11
```bash
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
python app.py
```

Or without activating the venv:
```bash
venv/bin/python app.py
```

> ⏳ **First run takes 3–10 minutes** — the CNN trains on MNIST before the window opens.

---

## 🪟 Setup on Windows

### 1. Install Python 3.11 from python.org
Download from: https://www.python.org/downloads/release/python-3110/

Make sure to check **"Add Python to PATH"** during installation.

### 2. Clone or download the project
```cmd
git clone <your-repo-url>
cd Learn2Count
```

### 3. Create a virtual environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies
```cmd
pip install -r requirements.txt
```

### 5. Run the app
```cmd
python app.py
```

> ⏳ **First run takes 3–10 minutes** — the CNN trains on MNIST before the window opens.

---

## 🎮 Controls

| Action | Result |
|---|---|
| **Left-click + drag** | Draw a digit on the canvas |
| **Right-click** | Predict the drawn digit (result in terminal) |
| **Press `c`** | Clear the canvas without predicting |
| **Press `q` or `ESC`** | Quit the application |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | CNN model training & inference |
| `opencv-python` | Drawing canvas & image processing |
| `numpy` | Array & image manipulation |
| `matplotlib` | Available for plotting if needed |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
Learn2Count/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── venv/                   # Virtual environment (not tracked in git)
└── handwritten_cnn.keras   # Saved model (auto-generated on first run, not tracked in git)
```

---

## 🐛 Troubleshooting

### `mutex lock failed: Invalid argument` (macOS)
You are using the system Python (`/usr/bin/python3`). Use Homebrew Python 3.11 instead:
```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### `ModuleNotFoundError: No module named 'tensorflow'`
```bash
pip install tensorflow
```

### Window doesn't open on macOS
Make sure you are **not** inside a headless environment (e.g., SSH). OpenCV requires a display. Run it directly in Terminal or iTerm2.

### Slow training
Training 10 epochs on CPU is expected to take **3–10 minutes**. This only happens on the **first run** — after that the model is loaded from `handwritten_cnn.keras`.

---

## 📄 License

MIT License — free to use and modify.
