# Egsi
This project implements a real-time sign language recognition and translation system using an infrared (IR) camera. The IR camera can reliably detect hand movements even in low-light or disaster situations, ensuring high reliability under various lighting and environmental conditions. The system utilizes MediaPipe for hand landmark extraction and a trained MLP model for recognition, delivering results in real time through both voice output and on-screen display.



![Image](https://github.com/user-attachments/assets/9ea9e5ba-af44-4d18-8904-0ffd656b4c94)

## Usage

### Installation


This codebase was developed and tested with the following packages.

- OS: Ubuntu 20.04.6 LTS
- CUDA: 12.9
- PyTorch: 2.7.1
- Python: 3.13.5

```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu129
pip install -r .\requirements.txt
```

### Dataset
The dataset was constructed based on coordinate features extracted from hand landmarks, which were used as direct inputs for model training.
```
<data>
|-- <hand_sign_data>
|-- <hand_sign_binary.pth>
```
### Training
Uses an IR camera and MediaPipe Hands to extract 63 hand landmark coordinates and a label, and saves them to hand_sign_data.csv.
Labels are set as `0 = None`, `1 = I am sick`, and `2 = Help me please`.

`traim.py` uses an IR camera and MediaPipe Hands to extract 63 hand landmark coordinates and a label, and saves them to `hand_sign_data.csv`.

Trains an MLP model using the landmark data from hand_sign_data.csv and saves the best-performing model as `hand_sign_binary.pth`
```
python make.py
python train.py
```
### Inference
This script uses a trained MLP model (.pth) and an IR camera
to recognize hand gestures in real time, and provides voice guidance (TTS) and on-screen display when the specified conditions are met.
```
python Egsi.py
```
