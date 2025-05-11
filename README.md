
# 🧠 AI Project - Fashion MNIST RGB Classifier

## 📌 Overview
This project builds and uses a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset after converting them to RGB format. It includes:
- A Jupyter Notebook for training and evaluation
- A standalone Python script for making predictions
- A pre-trained model and associated label mappings

## 📁 Project Structure

```
ai project/
├── Untitled24.ipynb           # Model training & evaluation notebook
├── ap-2.py                    # Inference script for predictions
├── fashion_mnist_rgb_fast.h5 # Trained CNN model
├── class_names.pkl            # Class label names (pickled)
```

## ⚙️ Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib (optional, for plots)

Install dependencies with:

```bash
pip install tensorflow numpy matplotlib
```

## 🚀 Usage

### 📓 Run the Notebook
Use Jupyter to explore model training and evaluation:
```bash
jupyter notebook Untitled24.ipynb
```

### 🧪 Run the Inference Script
Make predictions using the trained model:
```bash
python ap-2.py
```
Ensure `fashion_mnist_rgb_fast.h5` and `class_names.pkl` are in the same directory as `ap-2.py`.

## 📄 License
This project is provided for educational purposes. Modify and use it at your own discretion.

## 🙋‍♂️ Author
Made by [Your Name]. Contributions welcome!
