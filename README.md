# Automated Plant Disease Detection Using Deep Learning and Transfer Learning

This project detects plant leaf diseases using a Deep Learning CNN model trained on the PlantVillage Dataset. It supports common crops such as Apple, Grape, Corn, Tomato, Potato, etc. A ResNet50 Transfer Learning Model is trained and fine-tuned for high accuracy, and then deployed for real-time image prediction with Grad-CAM heatmap

**Link**: https://automated-plant-disease-detection.streamlit.app/

---

## Problem Statement
Plant diseases cause serious yield losses worldwide. Traditional disease detection is:

- Time-consuming  
- Requires expert presence  
- Error-prone  
- Not scalable for large farms  

This project builds an AI system that detects plant diseases using leaf images — fast, accurate, and scalable.

## Project Overview
This project uses **deep learning + transfer learning (ResNet50)** to classify **38 plant disease categories**, including healthy plants.

### Key Features
- Leaf image upload & preprocessing  
- Automatic disease prediction  
- Grad-CAM region highlighting  
- AI care recommendations (Hugging Face LLM)  
- Stylish Streamlit UI  
- Cloud deployment ready  

## Tech Stack

### Core
- Python 3.8+
- TensorFlow 2.x, Keras
- Streamlit
- OpenCV, NumPy, Pandas
- Matplotlib, Seaborn

### AI Models
- **ResNet50** (Transfer learning)
- **Mistral-7B-Instruct** (HuggingFace — plant care suggestions)

### Tools
- Jupyter Notebook
- Git & GitHub
- Kaggle Dataset

## Dataset
### PlantVillage Dataset
- Source: Kaggle
- ~54,000 images
- 38 classes (multiple crop types & diseases)
- JPG/PNG, RGB images
- Class imbalance addressed using augmentation


## Approach & Methodology

### Data Preprocessing
- Resize → 224×224
- Normalization
- Train/Validation/Test = 70/15/15
- Stratified sampling

### Data Augmentation
- Random flips, rotation, zoom
- Brightness / contrast shifts

### Model Development
| Stage | Method | Purpose |
|-------|--------|---------|
| Phase-1 | Custom CNN | Baseline |
| Phase-2 | ResNet50 pretrained | High feature specialization |
| Phase-3 | Layer unfreezing | Domain adaptation |

### Evaluation
- Accuracy, confusion matrix
- Grad-CAM interpretability

## Implementation

### Model Architecture (ResNet50-based)
```python
inputs = Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(38, activation='softmax')(x)
```

### Training Strategy

| Phase | Trainable Layers | Learning Rate | Goal | Outcome |
|-------|------------------|---------------|------|---------|
| Phase-1 | Frozen base model | 1e-4 | Fast learning | Removed Overfitting |
| Phase-2 | Last 30 layers unfrozen | 1e-5 | Fine-tuning | +2% accuracy gain |

### Additional Engineering

- Optimizer → Adam
- Loss → SparseCategoricalCrossentropy
- Regularization → Dropout + Data Augmentation
- Callbacks → EarlyStopping + ReduceLROnPlateau

### Evaluation & Visualization

- Accuracy, Loss trends
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- Grad-CAM heatmaps for explainable AI

## Results

| Model | Training | Validation | Test |
|-------|----------|------------|------|
| Baseline CNN | 83% | 77% | 79% |
| ResNet50 Frozen | 70% | 89% | 85% |
| ResNet50 Fine-Tuned | 83% | 94% | 94% |

### Observations:

- Excellent generalization — no major gap between Train & Test
- Strong class separation confirmed by Grad-CAM and confusion matrix
- Real-time performance achieved even without GPU

## Live Application

**Link**: https://automated-plant-disease-detection.streamlit.app/

## Limitations & Challenges

- Dataset: 
   * Only PlantVillage classes
   * No real-world noise (blur, lighting, multiple leaves)

- Model:
   * Overconfident on out-of-distribution images
   * No early uncertainty indicator

- Deployment
   * Streamlit RAM constraints
   * Model loading time depends on cold start
 
## Future Improvements

- Confidence calibration (Temperature Scaling)
- TFLite quantization for mobile use
- Add more crops and multiple co-occurring diseases
- Offline prediction app with camera scanning
- Disease treatment roadmap with severity scoring

## Contact

- GitHub → [@Ishaaq09](https://github.com/Ishaaq09)
- Project → https://github.com/Ishaaq09/Automated_plant_disease_detection_using_Deep_Learning_and_Transfer_Learning
