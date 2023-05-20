# handTracker-py

A simple hand tracker using MediaPipe and OpenCV

## Usage

### Including repo

To add the repo in another repo as submodule, run the following command

```shell
git submodule add https://github.com/AspectOfJerry/handTracker_py.git
```

### Installing dependencies

Run the following command in the root folder of the project:

```shell
pip install -r requirements.txt
```

### Importing

```python
from handTracker_py.handTracker import handTracker
```

### Documentation

#### handDetector()

```python
# leave `mode` True for video streams (false for stream of unrelated images), maxHands, detectionConfidence, trackingConfidence
def __init__(self, mode=True, maxHands=2, detectionConfidence=0.55, trackingConfidence=0.5)
```

Invoke it like this:

```python
tracker = handDetector()
```

#### findHands()

```python
# image, draw on the image?
def findHands(self, image, draw=True)
# returns new image
```

#### getPos()

```python
# image, hand number, draw on the image?
def getPos(self, image, handN=0, draw=True)
# returns array of the landmarks
```

Put the result in an array and access each landmark by index. Refer to the image below for more detail on each id.

### Landmarks

![Landmarks](./mediapipe_hand_landmarks.png)
