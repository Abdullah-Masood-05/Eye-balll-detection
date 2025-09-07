# Eye Ball Detection & Gaze Tracking

This project uses OpenCV and MediaPipe to detect eye landmarks and estimate gaze direction from a video file.

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd Eye-balll-detection
```

### 2. Create and Activate a Virtual Environment

**Windows**:

```sh
python -m venv env
env\Scripts\activate
```

### 3. Install Required Libraries

Run the following command to install dependencies from `requirements.txt`:

```sh
pip install -r requirements.txt
```

### How to Run

* **Place your input video file** (e.g., `eye_ball_2.mp4`) in the project directory.

* **Open** `eye_gaze_video.py`.

* **Edit the code to set your input filename**:

  In `eye_gaze_video.py`, change the line:

  ```sh
  video_path = "eye_ball_2.mp4"
  ```

---