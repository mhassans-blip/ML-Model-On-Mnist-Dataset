# MNIST ANN Deployment

This project deploys the ANN from `mnist_classification.ipynb` using:

- TensorFlow/Keras for the ANN
- FastAPI for the prediction API
- Streamlit for the frontend

## 1. ANN used in the class notebook

The original model is kept unchanged:

```text
Input: 28 × 28
      ↓
Flatten
      ↓
Dense(128, ReLU)
      ↓
Dense(32, ReLU)
      ↓
Dense(10, Softmax)
```

The notebook uses:

- MNIST dataset
- Pixel normalization: `X_train / 255` and `X_test / 255`
- `sparse_categorical_crossentropy`
- Adam optimizer
- 25 epochs
- 20% validation split

## 2. Train and save the model

Open:

```text
mnist_classification_deployment.ipynb
```

Run all cells.

The final cell saves:

```text
mnist_model.keras
```

IMPORTANT: Keep `mnist_model.keras` in this project folder.

## 3. Install packages

Open a terminal in this folder:

```bash
pip install -r requirements.txt
```

## 4. Start FastAPI

Run:

```bash
uvicorn api:app --reload
```

FastAPI will run at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## 5. Start Streamlit

Open a SECOND terminal in the same folder:

```bash
streamlit run app.py
```

Streamlit will open in your browser.

## 6. How it works

```text
User uploads digit
        ↓
Streamlit frontend
        ↓
POST /predict
        ↓
FastAPI
        ↓
Image preprocessing
        ↓
MNIST ANN model
        ↓
Prediction probabilities
        ↓
Predicted digit + confidence
        ↓
Streamlit displays result
```

## 7. Important image note

The MNIST dataset normally represents a bright handwritten digit on a dark background.

The API automatically checks the image brightness and inverts it when the uploaded image appears to have a light background. This makes ordinary handwritten digit images easier to use.

## 8. API test

After starting FastAPI, open:

```text
http://127.0.0.1:8000/docs
```

Use the `/predict` endpoint and upload an image.

The API returns:

```json
{
  "prediction": 7,
  "confidence": 0.98,
  "probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    "2": 0.001,
    "3": 0.002,
    "4": 0.0001,
    "5": 0.001,
    "6": 0.0001,
    "7": 0.98,
    "8": 0.01,
    "9": 0.005
  }
}
```

The numbers above are only an example. Actual probabilities depend on the uploaded image.

## 9. Project files

```text
mnist_ann_deployment/
│
├── mnist_classification.ipynb
├── mnist_classification_deployment.ipynb
├── mnist_model.keras       ← generated after training
├── api.py
├── app.py
├── requirements.txt
└── README.md
```
