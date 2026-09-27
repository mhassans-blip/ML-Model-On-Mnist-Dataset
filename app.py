import streamlit as st
import requests
from PIL import Image
import pandas as pd
import uvicorn

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Digit Recognition")
st.write("Upload a handwritten digit and let the trained ANN predict it.")

st.info(
    "Your class ANN is used here: Flatten → Dense(128) → Dense(32) → Dense(10)."
)

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    if st.button("Predict Digit", type="primary"):
        with st.spinner("Sending image to FastAPI..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    digit = result["prediction"]
                    confidence = result["confidence"]

                    st.success(f"Predicted Digit: {digit}")
                    st.metric("Confidence", f"{confidence:.2%}")

                    st.subheader("Prediction Probabilities")

                    probabilities = result["probabilities"]

                    df = pd.DataFrame({
                        "Digit": list(probabilities.keys()),
                        "Probability": list(probabilities.values())
                    })

                    st.bar_chart(
                        df.set_index("Digit")
                    )

                else:
                    try:
                        error = response.json().get("detail", response.text)
                    except Exception:
                        error = response.text

                    st.error(f"API Error: {error}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "FastAPI is not running. Start it first with: "
                    "uvicorn api:app --reload"
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.divider()
st.caption("MNIST ANN Deployment • FastAPI + Streamlit")
