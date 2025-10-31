# Bilingual Sign Language Translation (English & Arabic)

This project focuses on **real-time bilingual sign language translation** (English and Arabic) using **Deep Learning (CNN-based models)** and **Computer Vision**. It aims to bridge the communication gap between sign language users and non-signers by detecting and translating sign gestures into text.

---

## Project Overview

The system is built in two main stages:

1. **Preprocessing & Dataset Preparation**

   * Images of hand gestures (A–Z, Space, Backspace) were collected and preprocessed for training.
   * All images were resized to 224×224, normalized, and augmented to increase diversity.
   * The final dataset contains ~57,000 images across 28 classes.

2. **Model Training (CNN)**

   * The processed dataset (`asl_cnn_dataset.h5`) can be directly used to train a **Convolutional Neural Network (CNN)** for gesture recognition.
   * The model outputs alphabet predictions, which can then be converted into text in real time.

---

## Dataset

The preprocessed dataset is available on Google Drive:
  [**Download Preprocessed ASL Dataset**](https://drive.google.com/file/d/1pr9Y90AI00PdY9oZG-VUh4vgTb3g0aUw/view?usp=drive_link)

Files included:

* `asl_cnn_dataset.h5` — the main dataset file for CNN training
* `label_encoder.pkl` — used to map class labels (A–Z, Space, Backspace)
* `class_distribution_cnn.png` — shows class balance after preprocessing

---

## Technologies Used

* **Python 3.11+**
* **TensorFlow / Keras** (for CNN model development)
* **OpenCV** (for image preprocessing)
* **NumPy & scikit-learn** (for data handling and encoding)
* **Matplotlib** (for visualizations)
* **h5py** (for saving and reading large HDF5 datasets)

---

## Related Work — ARSL (Arabic Sign Language Translator)

This project was inspired by and closely related to **ARSL (Arabic Sign Language Translator)**, which uses similar techniques such as CNNs and computer vision for gesture detection.
Both systems share a common goal: enabling **bilingual communication** (English and Arabic) through **AI-driven sign language recognition**.
While ARSL focuses primarily on Arabic gestures, this project extends the idea to **English Sign Language (ASL)**, aiming for future **bilingual integration**.

---

## How to Use

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/Bilingual-Sign-Language-Translation.git
   cd Bilingual-Sign-Language-Translation
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and place the dataset (`asl_cnn_dataset.h5` and `label_encoder.pkl`) in the project root directory.

4. Train the CNN model:

   ```bash
   python train_cnn.py
   ```

5. Once trained, the model can be used in the live translation module to recognize signs via webcam.

---

## Data Summary

| Metric                            | Value                      |
| :-------------------------------- | :------------------------- |
| Total Images                      | ~57,000                    |
| Classes                           | 28 (A–Z, Space, Backspace) |
| Missing Values                    | 0%                         |
| Data Accuracy After Preprocessing | ~100%                      |
| Image Size                        | 224 × 224 pixels           |
| File Format                       | HDF5 (.h5)                 |

---

## Future Improvements

* Add **real-time hand tracking** for smoother live predictions.
* Extend dataset for **Arabic signs** to complete bilingual coverage.
* Integrate a **Transformer-based text translation module** for multilingual output.

