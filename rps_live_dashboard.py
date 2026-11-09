from pathlib import Path
import math
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

# --------------------------------------------------------
# 1. FIND FILES AND DEFINE CLASS LABELS
# --------------------------------------------------------
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "rps_model.pth"
HAND_MODEL_PATH = ROOT / "hand_landmarker.task"
CLASS_NAMES = ["ROCK", "PAPER", "SCISSORS"]  # Same order as training!

# --------------------------------------------------------
# 2. DEFINE BOTH VERSIONS OF THE NETWORK
# The weights file contains clues about whether ReLU layers
# were present (Sequential layer indices differ).
# --------------------------------------------------------
class RPSNetwork(torch.nn.Module):
    def __init__(self, use_relu: bool):
        super().__init__()
        if use_relu:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(63, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 3),
            )
        else:
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(63, 32),
                torch.nn.Linear(32, 16),
                torch.nn.Linear(16, 3),
            )

    def forward(self, x):
        return self.layers(x)


def load_trained_network():
    """Load the trained weights, not a random untrained model."""
    # Only load model files that you created or trust.
    weights = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    keys = set(weights)
    if "layers.4.weight" in keys:
        use_relu = True
    elif "layers.2.weight" in keys:
        use_relu = False
    else:
        raise ValueError("Unrecognized checkpoint architecture. Re-run train_network.")
    network = RPSNetwork(use_relu)
    network.load_state_dict(weights)  # Refuses mismatched layer shapes/keys.
    network.eval()  # Inference mode, not training.
    return network, use_relu


# --------------------------------------------------------
# 3. EXACT SAME NORMALIZATION AS THE DATA COLLECTOR
# Do not change these formulas without retraining!
# --------------------------------------------------------
def normalize_landmarks(landmarks, width, height):
    wrist = landmarks[0]        # Landmark 0 = wrist
    middle_base = landmarks[9]  # Landmark 9 = middle-finger base

    scale = math.hypot(
        (middle_base.x - wrist.x) * width,
        (middle_base.y - wrist.y) * height,
    )
    if scale < 1:
        return None

    features = []
    for p in landmarks:
        x = (p.x - wrist.x) * width / scale
        y = (p.y - wrist.y) * height / scale
        z = (p.z - wrist.z) * width / scale
        features.extend([x, y, z])
    return features  # Exactly 63 numbers, in x0,y0,z0,... order.


# --------------------------------------------------------
# 4. GET INTERMEDIATE ACTIVATIONS (FOR VISUALIZATION)
# Unlike the final prediction, these are hidden-neuron outputs.
# --------------------------------------------------------
def infer_and_inspect(model, features, use_relu):
    x = torch.tensor([features], dtype=torch.float32)  # Shape: [1,63]
    with torch.no_grad():
        first_linear = model.layers[0](x)    # Shape: [1,32]
        if use_relu:
            hidden1 = model.layers[1](first_linear)
            second_linear = model.layers[2](hidden1)
            hidden2 = model.layers[3](second_linear)
            logits = model.layers[4](hidden2)
        else:
            hidden1 = first_linear
            hidden2 = model.layers[1](hidden1)
            logits = model.layers[2](hidden2)
        probabilities = torch.softmax(logits, dim=1)[0]

    return (
        hidden1[0].numpy(),
        hidden2[0].numpy(),
        logits[0].numpy(),
        probabilities.numpy(),
    )


# --------------------------------------------------------
# 5. DRAWING HELPERS
# OpenCV uses BGR colors rather than RGB.
# --------------------------------------------------------
BG = (25, 23, 18)
PANEL = (42, 39, 33)
WHITE = (240, 240, 236)
MUTED = (165, 170, 170)
CYAN = (226, 207, 89)
GREEN = (140, 211, 114)
ORANGE = (75, 165, 245)


def label(canvas, text, x, y, size=0.56, color=WHITE, thickness=1):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thickness, cv2.LINE_AA)


def make_dashboard(frame, landmarks, features, activations, use_relu,
                   show_numbers, show_activations):
    """Build one 1280x720 image containing camera + network internals."""
    canvas = np.full((720, 1280, 3), BG, dtype=np.uint8)
    cv2.rectangle(canvas, (12, 12), (1268, 70), PANEL, -1)
    label(canvas, "RPS  /  NEURAL NETWORK LAB", 30, 49, .9, WHITE, 2)
    label(canvas, "TRAINED MODEL - LIVE INFERENCE", 905, 47, .52, CYAN)

    # Left: webcam with hand skeleton.
    view = cv2.resize(frame, (640, 480))
    h0, w0 = frame.shape[:2]
    if landmarks:
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        for c in connections:
            a = landmarks[c.start]
            b = landmarks[c.end]
            cv2.line(view, (int(a.x * 640), int(a.y * 480)),
                     (int(b.x * 640), int(b.y * 480)), GREEN, 2, cv2.LINE_AA)
        for i, p in enumerate(landmarks):
            xx, yy = int(p.x * 640), int(p.y * 480)
            cv2.circle(view, (xx, yy), 4, ORANGE, -1, cv2.LINE_AA)
            if show_numbers:
                label(view, str(i), xx + 5, yy - 4, .37, WHITE)
    canvas[92:572, 20:660] = view
    label(canvas, "CAMERA + 21 HAND LANDMARKS", 25, 601, .53, WHITE)
    label(canvas, "63 INPUTS = 21 JOINTS x (X, Y, Z)", 25, 628, .51, MUTED)
    label(canvas, "Wrist-relative, hand-size normalized", 25, 652, .47, MUTED)

    # Right: probabilities and hidden layer activations.
    cv2.rectangle(canvas, (680, 92), (1260, 572), PANEL, -1)
    label(canvas, "NETWORK: 63 > 32 > 16 > 3", 700, 124, .64, WHITE, 2)
    label(canvas, "Activation: " + ("ReLU" if use_relu else "None (linear)"),
          700, 150, .50, CYAN)

    if activations is None:
        label(canvas, "NO HAND DETECTED", 720, 275, .9, ORANGE, 2)
        label(canvas, "Show your hand to see the neurons fire.", 720, 308, .51, MUTED)
    else:
        h1, h2, logits, probs = activations
        label(canvas, "OUTPUT PROBABILITIES (SOFTMAX)", 700, 191, .52, MUTED)
        best = int(np.argmax(probs))
        for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
            top = 217 + 49 * i
            label(canvas, name, 700, top + 15, .53, WHITE)
            cv2.rectangle(canvas, (826, top), (1158, top + 20), (65, 65, 65), -1)
            color = GREEN if i == best else CYAN
            cv2.rectangle(canvas, (826, top), (826 + int(332 * float(p)), top + 20), color, -1)
            label(canvas, f"{float(p) * 100:5.1f}%", 1166, top + 16, .49, WHITE)
        label(canvas, "Raw scores (logits): " + ", ".join(f"{v:+.2f}" for v in logits),
              700, 384, .43, MUTED)

        if show_activations:
            label(canvas, "HIDDEN NEURON OUTPUTS", 700, 425, .50, MUTED)
            # 32 first-layer values and 16 second-layer values. Each
            # small bar shows magnitude, not a probability.
            for layer_idx, values in enumerate((h1, h2)):
                y0 = 442 + layer_idx * 52
                label(canvas, f"L{layer_idx+1}", 700, y0 + 14, .46, WHITE)
                max_abs = max(float(np.max(np.abs(values))), 1e-6)
                bw = 13 if layer_idx == 0 else 25
                spacing = 2 if layer_idx == 0 else 5
                for j, value in enumerate(values):
                    x0 = 743 + j * (bw + spacing)
                    length = int(25 * abs(float(value)) / max_abs)
                    color = GREEN if value >= 0 else ORANGE
                    cv2.rectangle(canvas, (x0, y0 + 28 - length),
                                  (x0 + bw, y0 + 28), color, -1)
        else:
            label(canvas, "Hidden activations hidden - press H", 700, 461, .49, MUTED)

    label(canvas, "Q quit  |  N landmark numbers  |  H neuron bars", 25, 690, .52, WHITE)
    label(canvas, "Predictions are estimates, not guaranteed probabilities.",
          702, 691, .44, MUTED)
    return canvas


# --------------------------------------------------------
# 6. MAIN LOOP: CAMERA -> LANDMARKS -> NETWORK -> DISPLAY
# --------------------------------------------------------
def main():
    for path in (MODEL_PATH, HAND_MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name} in {ROOT}")

    model, use_relu = load_trained_network()
    print("Loaded model:", "WITH ReLU" if use_relu else "WITHOUT ReLU")
    print("Model:", MODEL_PATH)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam. Close any other program using it.")

    start = time.monotonic()
    last_timestamp = -1
    show_numbers = False
    show_activations = True

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # Match the data collector's mirroring.
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(int((time.monotonic()-start)*1000), last_timestamp+1)
                last_timestamp = timestamp
                result = detector.detect_for_video(mp_image, timestamp)

                landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
                features = normalize_landmarks(landmarks, w, h) if landmarks else None
                activations = (infer_and_inspect(model, features, use_relu)
                               if features is not None else None)
                screen = make_dashboard(frame, landmarks, features, activations,
                                        use_relu, show_numbers, show_activations)
                cv2.imshow("RPS / Neural Network Lab", screen)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("n"):
                    show_numbers = not show_numbers
                elif key == ord("h"):
                    show_activations = not show_activations
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
