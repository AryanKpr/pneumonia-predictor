# Pneumonia-detector
# Pneumonia Detection from Chest X-Rays
Deep learning model to detect pneumonia from chest X-ray images using PyTorch and Streamlit.
## Results
- Overall Accuracy: 95%
- Sensitivity (Pneumonia Detection): 97.8%
- Specificity (Normal Detection): 89.5%
- 
## Try It Out

[Live Demo on Streamlit Cloud]https://pneumo-vision-ai.streamlit.app/

## Tech Stack

- Deep Learning: PyTorch
- Frontend: Streamlit

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle
- Training: 5,216 images (1,341 Normal, 3,875 Pneumonia)
- Test: 624 images

## Model Architecture

Custom CNN:
- 2 Convolutional layers (6 and 16 filters)
- Max pooling layers
- 3 Fully connected layers
- Input: 64x64 grayscale images
