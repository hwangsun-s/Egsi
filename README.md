# Egsi
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
To add additional sign gestures, each gesture was recorded in real time using a camera, the LABEL in `make.py` was updated to the corresponding gesture class, and the model was retrained.
```
python make.py
python train.py
```
### Inference
Run the model
```
python Egsi.py
```
