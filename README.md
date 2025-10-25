# Eye‑Ball Detection & Gaze Tracking

A lightweight **pupil‑based gaze tracker** that uses OpenCV and MediaPipe to detect the white part of the eye, estimate the pupil (iris) center, and infer the user’s gaze direction in real time.

> **What’s new?**  
> * Simplified concept – the pupil is tracked *inside* the white eye area.  
> * Intuitive visual feedback (diagrams, colored markers, direction text).  
> * Supports combinations of directions (e.g., **UP‑LEFT**, **DOWN‑RIGHT**).  
> * Optional smoothing to reduce jitter.

---

## Setup

> **Prerequisites** – Python 3.8+ and `pip`.

### 1. Clone the repository

```bash
git clone https://github.com/Abdullah-Masood-05/Eye-balll-detection.git
cd Eye-balll-detection
```

> The repo was originally named `Eye-balll-detection`; the new code lives in the same folder.

### 2. Create & activate a virtual environment

**Windows**

```bash
python -m venv env
env\Scripts\activate
```

**macOS / Linux**

```bash
python -m venv env
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The `requirements.txt` file contains:
> ```
> opencv-python
> mediapipe
> numpy
> ```

---

## Running the Tracker

1. **Place an input video** (e.g., `eye_ball_2.mp4`) in the project root.

2. **Open `pupil_gaze_tracker.py`** – the main script that implements the new pupil‑based logic.

3. **Edit the input file name** at the top of the script:

   ```python
   video_path = "eye_ball_2.mp4"   # <-- change this to your video
   ```

4. **Execute**

   ```bash
   python pupil_gaze_tracker.py
   ```

   *If you still prefer the old landmark‑based tracker, you can keep running `eye_gaze_video.py` – it will automatically fall back to the new algorithm.*

---

## How the New Tracker Works

| Step | What’s Done | Why it matters |
|------|-------------|----------------|
| 1️⃣ | **Eye boundary detection** (white sclera) | Gives a stable reference for the eye. |
| 2️⃣ | **Iris center estimation** (pupil) | The core of the new approach – the black part is the cue. |
| 3️⃣ | **Relative position** (`dx`, `dy` from eye center) | Maps the pupil’s offset to left/right & up/down. |
| 4️⃣ | **Direction inference** (threshold‑based) | Produces *LEFT*, *RIGHT*, *UP*, *DOWN*, *CENTER*, or a two‑word combo. |
| 5️⃣ | **Optional smoothing** | Reduces rapid flips in the output. |

The visual overlay contains:

| Element | Color | Meaning |
|---------|-------|---------|
| Green box | `#00FF00` | Eye outline with a center marker. |
| Yellow circle | `#FFFF00` | Current pupil location. |
| Magenta line | `#FF00FF` | Vector from eye center to pupil. |
| Small white eye diagram | — | Shows the pupil’s position relative to the eye center. |
| Large gaze direction text | — | Printed at the bottom of the frame. |

---

## Example Output

```
Gaze Direction: RIGHT
```

When you run the script you’ll see a video window with the overlay described above, and the direction text updates live as the pupil moves.

---

## Customizing

| Variable | Default | What it changes |
|----------|---------|-----------------|
| `pupil_threshold` | 0.25 | Minimum normalized offset (in eye‑center units) required to declare a movement. |
| `smoothing_factor` | 0.2 | Exponential smoothing coefficient (0 = no smoothing, 1 = full smoothing). |
| `output_to_console` | `True` | Prints the direction to the terminal for debugging. |

You can edit these values near the top of `pupil_gaze_tracker.py`.

---

## Known Issues & Future Work

* The current iris center detector is a simple circular Hough approach – more robust algorithms (e.g., blob detection or deep‑learning based pupil detection) are on the roadmap.  
* Works best on frontal‑face videos with good lighting; extreme head poses may reduce accuracy.  

Feel free to open an issue or PR if you spot a bug or have a feature idea!
