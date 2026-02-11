# Object Removal

Object Removal From Image using Deep Learning (DeepFillv2 GAN-based Image Inpainting).

This project allows users to manually select an object in an image and remove it using a GAN-based inpainting model that reconstructs the missing region with semantically consistent content.

---

## Project Overview

- Model: DeepFillv2  
- Framework: PyTorch  
- Architecture: GAN-based Image Inpainting  
- Interface: PyQt GUI  
- GPU Support: Yes (CUDA if available)  

---

## Setup & Run

### 1️⃣ Clone the Repository

```
git clone https://github.com/DsThakurRawat/object-Removal.git
cd object-Removal
```

---

### 2️⃣ (Recommended) Create Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Requirements

```
pip3 install -r requirements.txt
```

---

### 4️⃣ Download Pretrained Model

The DeepFillv2 model requires pretrained weights.

Download from:  
https://drive.google.com/u/0/uc?id=1L63oBNVgz7xSb_3hGbUdkYW1IuRgMkCa&export=download

Place the downloaded `.pth` file inside:

```
src/models/
```

---

### 5️⃣ Run the Application

```
python3 src/app.py
```

---

## How to Use

- Browse and select image:

<p align="center">
  <img src="/1.png" width="1000" />
</p>

- Select the desired object which needs to be removed:

<p align="center">
  <img src="/2.png" width="1000" />
</p>

- Press **"Enter"** to remove object.  
- Press **"R"** to reset selection.

<p align="center">
  <img src="/3.png" width="1000" />
</p>

---

## Results

### Input Image

<p align="center">
  <img src="/ip.jpeg" width="1000" />
</p>

### Output Image

<p align="center">
  <img src="/op.png" width="1000" />
</p>

---

## Project Structure

```
ObjectRemoval-main/
│
├── src/
│   ├── app.py
│   ├── objRemove.py
│   └── models/
│       └── (place pretrained .pth file here)
│
├── 1.png
├── 2.png
├── 3.png
├── ip.jpeg
├── op.png
├── requirements.txt
└── README.md
```

---

## Model Information

- Model: DeepFillv2  
- Framework: PyTorch  
- Architecture: Gated Convolution GAN  
- Supports GPU acceleration (CUDA if available)  

---

## Dependencies

- Python 3.8+
- OpenCV (cv2)
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- PyQt

---

## Limitations

- Performance depends on accurate mask selection  
- Large complex regions may produce artifacts  
- Requires pretrained model weights  
- CPU execution may be slower for high-resolution images  

---

## Future Improvements

- Automatic object detection + mask generation  
- Convert to web application (FastAPI / Streamlit)  
- Docker support for reproducibility  
- REST API integration  
- Batch image processing  
- Automatic model weight download  

---

## Reference

Yu, Jiahui, et al.  
**"Free-Form Image Inpainting with Gated Convolution."**  
ICCV 2019-2020.

---

## License

This project is licensed under the MIT License.
