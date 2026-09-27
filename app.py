import streamlit as st
import requests
from PIL import Image
import pandas as pd
import subprocess
import time


# --------------------------------
# Start FastAPI server
# --------------------------------

@st.cache_resource
def start_fastapi():

    process = subprocess.Popen([
        "uvicorn",
        "api:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
    ])

    time.sleep(5)

    return process


fastapi_process = start_fastapi()

API_URL = "http://127.0.0.1:8000"


# --------------------------------
# Streamlit UI
# --------------------------------

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Digit Recognition")

st.write(
    "Upload a handwritten digit and let the trained ANN predict it."
)

st.info(
    "ANN Architecture: "
    "Flatten → Dense(128) → Dense(32) → Dense(10)"
)


# --------------------------------
# Check FastAPI
# --------------------------------

try:

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=10
    )

    if health_response.status_code == 200:

        health = health_response.json()

        if health.get("model_loaded"):

            st.success("FastAPI + MNIST Model is running ✅")

        else:

            st.error(
                f"Model Error: {health.get('error')}"
            )

except Exception as e:

    st.error(
        f"FastAPI could not start: {e}"
    )


# --------------------------------
# Upload image
# --------------------------------

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


    if st.button(
        "Predict Digit",
        type="primary"
    ):

        with st.spinner(
            "FastAPI is processing the image..."
        ):

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
                    timeout=60
                )


                if response.status_code == 200:

                    result = response.json()

                    digit = result["prediction"]

                    confidence = result["confidence"]

                    st.success(
                        f"Predicted Digit: {digit}"
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2%}"
                    )


                    st.subheader(
                        "Prediction Probabilities"
                    )

                    probabilities = result[
                        "probabilities"
                    ]

                    df = pd.DataFrame({
                        "Digit": list(
                            probabilities.keys()
                        ),
                        "Probability": list(
                            probabilities.values()
                        )
                    })

                    st.bar_chart(
                        df.set_index("Digit")
                    )


                else:

                    try:

                        error = response.json().get(
                            "detail",
                            response.text
                        )

                    except Exception:

                        error = response.text

                    st.error(
                        f"API Error: {error}"
                    )


            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


st.divider()

st.caption(
    "MNIST ANN Deployment • "
    "FastAPI + Streamlit"
)
