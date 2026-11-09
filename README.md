# Rock Paper Scissors — Neural Network

I wanted to better understand how neural networks work so I built one combining two of my favourite topics (Computer Vision, and ML). I first Collected hand landmark data using MediaPipe and then trained a PyTorch neural network to classify Rock, Paper, and Scissors gestures in real time.

![Live RPS Neural Network Dashboard](media/NeuralNetworkExample.gif)

## 1. Turning a Hand Into Numbers

I used Mediapipes 21 landmarks to map the position of the 21 different points on my hand. Each point has 3 coordinates (x,y,z)

for each datapoint there is (21 landmarks x 3 coordinates each) =  
giving us 63 input features

I then normalized the coordinates relative to my wrist (landmark 0 on mediapipe) so the network focus's on the shape of my hand with the bias of where its position on the camera frame.

I collected 140 labeled training examples and 45 additional examples in a separate recording session for evaluation.

## 2. Building the Neural Network

I decided to design a neural network with 2 hidden layers

**63 inputs → 32 neurons → 16 neurons → 3 outputs**

The final layer produces three scores representing Rock, Paper, and Scissors. Using softmax, these scores can be converted into probabilities, and the gesture with the highest probability becomes the prediction.

The network contains 2,627 learnable parameters (weights and biases).

## 3. Teaching the Network to Learn

The neural network uses supervised learning with the 140 labelled data examples of rock paper and scissors hand positions.

The network starts with randomly initialized weights and makes a prediction and compares it against the correct gesture using a cross-entropy loss function. Backpropagation calculates how each weight affects the loss, and the Adam optimizer adjusts the weights to improve future predictions.

I repeated this process over 300 epochs, allowing the network to gradually learn patterns associated with each gesture.

## 4. Live Visualization

I used codex to build a live dashboard to visualize what happens inside the neural network.

The dashboard displays the webcam feed with MediaPipe landmarks, the predicted probabilities for each gesture, and the activations of neurons in the hidden layers.

## 5. Running the Project

### Requirements

- Python
- PyTorch
- MediaPipe
- OpenCV
- Webcam

### Step 1 — Clone the Repository

```bash
git clone https://github.com/panostzan/RPS-Neural-Network.git
cd RPS-Neural-Network
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

```bash
pip install torch mediapipe opencv-python
```

### Step 4 — Download the MediaPipe Model

Download the [MediaPipe Hand Landmarker model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) and place it in the main project directory as `hand_landmarker.task`.

### Step 5 — Collect Data

Run the data collection script to record labeled hand gestures, or use the included dataset.

### Step 6 — Train the Neural Network

Run the training script to train the model using your collected data and save the trained weights.

### Step 7 — Run the Live Dashboard

Run the live dashboard to see the model recognize Rock, Paper, and Scissors gestures in real time.
