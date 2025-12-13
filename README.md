# Object Removal
Object Removal From Image using DeepLearning.

## Run

- The DeepFillv2 model needs pretrained weights from [here](https://drive.google.com/u/0/uc?id=1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa&export=download) and put put the weights pth file in [src/models/](/src/models/).

- To install requirements for the project.

```
pip3 install -r requirements.txt
```

- To run the program, navigate to the project directory and execute the following command in the terminal:

```
python3 src/app.py
```

- Browse and select image

<p align ="center">
  <img src="/1.png" width="1000" />
  <em></em>
</p>

- Select the desired object which need to be removed

<p align ="center">
  <img src="/2.png" width="1000" />
  <em></em>
</p>

- Press "Enter" to remove object. (If you want to reset selection press "R")

<p align ="center">
  <img src="/3.png" width="1000" />
  <em></em>
</p>

## Results
Following are some results. 

<p align ="center">
  <img src="/ip.jpeg" width="1000" />
  <em></em>
</p>
<p align ="center">
  <img src="/op.png" width="1000" />
  <em></em>
</p>

## Dependencies
- python3
- cv2
- torchvision
- torch
- matplotlib
- numpy
- PyQt
# object-Removal
