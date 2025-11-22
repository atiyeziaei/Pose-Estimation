# Pose Estimation with MediaPipe

A real-time exercise tracking system using Google's MediaPipe for pose detection and joint angle calculation.

## Documentation

### MediaPipe Package
**MediaPipe** is an open-source framework developed by Google for building perception pipelines. It provides ready-to-use solutions for detecting and tracking human poses, hands, faces, and more.

- **Official Package Page**: https://mediapipe.dev/
- **Python Documentation**: https://google.github.io/mediapipe/solutions/pose.html
- **GitHub Repository**: https://github.com/google/mediapipe

MediaPipe uses advanced machine learning models optimized for real-time performance on both CPU and mobile devices.

## Project Overview

This notebook implements a real-time pose estimation pipeline for exercise tracking. The system:

1. **Detects Pose Landmarks** - Uses MediaPipe to identify 33 key points on the human body (shoulders, elbows, wrists, hips, knees, ankles, etc.)

2. **Calculates Joint Angles** - Computes angles between three consecutive joints using inverse tangent mathematics to determine body posture and movement

3. **Tracks Movement Stages** - Monitors transitions between exercise positions (e.g., "down" and "up" for bicep curls)

4. **Counts Repetitions** - Automatically counts completed exercise reps based on angle thresholds and movement patterns

5. **Real-time Visualization** - Overlays pose skeleton, joint angles, rep count, and movement stage on video feed

### Key Components

- **Pose Detection**: MediaPipe Pose model with configurable confidence thresholds
- **Angle Calculation**: Vector-based angle computation between body joints
- **Rep Counter**: State machine that tracks movement phases
- **Live Display**: OpenCV visualization with landmarks, angles, and statistics

### Workflow

```
Video Capture → Color Conversion → Pose Detection → Landmark Extraction 
→ Angle Calculation → Rep Logic → Visualization → Display
```

## Assignment Details

### Challenge

Implement one of the following exercises using MediaPipe's pose estimation capabilities. The top 3 submissions will receive prizes.

### Available Exercises

Choose and implement one of these movement types:

- Squat
- Deadlift
- Bench Press
- Overhead Press
- Pull-Up
- Push-Up
- Row
- Hip Hinge
- Lunge
- Plank
- Farmer Carry
- Sprint
- Dip
- Glute Bridge
- Step-Up
- Bulgarian Split Squat
- Kettlebell Swing
- Chest Fly
- Lat Pulldown
- Face Pull
- Hollow Hold
- Turkish Get-Up

### Submission Guidelines

1. **Fork the Repository** - Clone the class exercise repo to your account

2. **Implement Exercise** - Use the MediaPipe package to build detection and counting logic for your chosen exercise

3. **Polish the Product** - Design and refine the interface for production-quality output (UI/UX and functionality)

4. **Add Media** - Include demonstration photos and videos showcasing your implementation

5. **Organize Submission** - Create a dedicated folder containing:
   - Complete source code
   - Documentation
   - Demo images and videos
   - Any additional files
   - Place this folder alongside the main project

6. **Push to Repository** - Commit and push your complete submission to the exercise repo

### Evaluation Criteria

- Accuracy of pose detection and rep counting
- Code quality and documentation
- User interface design and usability
- Completeness of media documentation
- Production-readiness of the final solution

## Getting Started

### Installation

```bash
pip install opencv-python mediapipe numpy
```

### Running the Notebook

1. Open `Pose_Estimation_MediaPipe.ipynb` in Jupyter
2. Install dependencies in the Setup section
3. Run cells sequentially to explore pose detection
4. Adapt the final exercise counter cell for your chosen exercise

### Basic Usage

```python
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )
        
        cv2.imshow("Pose", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

## Key Concepts

### Pose Landmarks
MediaPipe detects 33 landmarks covering:
- Face (11 points)
- Upper body (9 points)
- Lower body (13 points)

Each landmark contains normalized coordinates (x, y, z) and visibility confidence.

### Angle Calculation
Angles are computed using the arctangent method for three consecutive joints:

```python
def calculate_angle(a, b, c):
    radians = np.arctan2(c.y-b.y, c.x-b.x) - np.arctan2(a.y-b.y, a.x-b.x)
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)
```

### Rep Counting Logic
Track movement stages using angle thresholds:
- When angle exceeds upper threshold → "down" stage
- When angle falls below lower threshold + previous stage was "down" → increment counter and set "up" stage

---

**Note**: This is an educational project demonstrating real-time pose estimation. For production use, consider optimization for different body types, exercise variations, and lighting conditions.
