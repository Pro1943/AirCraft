# **AIR CRAFT v0.1** 🧱🌬️

**“Build blocks in the air using your hands — no keyboard, no mouse, just gestures.”**


## **Features🔨**

* Hand 1 (H1) → Controls **cursor**
* Hand 2 (H2) → **Open → Close** triggers block placement
* **Fist** with H1 → Deletes nearest block
* Continuous placement is possible by repeating H2 clench
* Real-time webcam-based AR block building
* Minimalistic “holographic” block effect
* Lightweight: works on low-end PCs


## **Requirements📃**

### ⚠️Note: It is suggested that you make an vertual environment before installing the packages
* Python 3.12.12

* Packages:

  ```bash
  pip install -r requitments.txt
  ```

* Webcam (integrated or external)

* Hand Landmarker model: `hand_landmarker.task`
  *(download from [MediaPipe Hand Landmarker Docs](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker))*

## **Setup & Run🛞**

1. Clone or download **Air Craft** code
2. Make sure the model file `hand_landmarker.task` is in the same directory as the script
3. Install required packages (see **Requirements**)
4. Run:

```bash
python aircraft.py
```

5. ESC key → exit the program

## **Gestures & Controls👋**

| Hand         | Gesture                   | Action                     |
| ------------ | ------------------------- | -------------------------- |
| H1 (Cursor)  | Move hand                 | Move cursor                |
| H1 (Cursor)  | Fist (all fingers down)   | Delete nearest block       |
| H2 (Trigger) | Open → Close sequence     | Place a block at cursor    |
| H2           | Keep clenching repeatedly | Continuous block placement |



## **How it Works (In short)🚲**

1. **Webcam Feed** → Captured using OpenCV
2. **Hand Tracking** → MediaPipe HandLandmarker tracks landmarks
3. **Gesture Detection** → Python checks finger positions
4. **Blocks** → Stored as a list of `(x, y)` coordinates
5. **Drawing** → OpenCV draws “holographic” blocks on the frame
6. **Interaction** → Gestures manipulate block list in real time


## **Code Structure🖥️**

* `air_craft.py` → main program
* `hand_landmarker.task` → MediaPipe pre-trained model
* Blocks stored in `blocks = []`
* `is_hand_open()` → detects open hand for placement
* `get_nearest_block()` → finds block for deletion
* `draw_block()` → draws block on webcam frame


## **Next Steps / Future Ideas**

* 🎨 Colorful blocks
* 🔲 Grid system (like real Minecraft)
* 🔄 Rotate blocks with gestures
* 💾 Save/load “air worlds”
* 🎥 YouTube showcase: *“I built Minecraft in the air using Python & OpenCV”*

## **Tips for Best Experience**

* Well-lit room → better hand detection
* Keep webcam at chest/shoulder height
* Move slowly at first → MediaPipe is more stable
* Make gestures deliberate: open/close vs quick flicks


### **Disclaimer:⚠️⚠️**
This is a **fun prototype**, not a production-ready game. Performance may vary depending on PC and lighting conditions.

---