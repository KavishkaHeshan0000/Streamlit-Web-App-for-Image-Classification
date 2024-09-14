# Image Classification App

## Overview

This is a Streamlit-based web application that uses a pre-trained MobileNetV2 model from TensorFlow to classify images. Users can upload an image, adjust the confidence threshold for predictions, and view the results, including a bar chart of the top predictions.

## Features

- **Image Upload:** Upload images in JPG, JPEG, or PNG format.
- **Confidence Threshold:** Adjust the minimum confidence score to filter predictions.
- **Results Display:** View classification results and a bar chart of the top 10 predictions.

## How to Run

### Streamlit App

You can access the live Streamlit app 👉[here](https://your-streamlit-app-link)👈. 

### Local Setup

To run the app locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/KavishkaHeshan0000/Image_Classification_WebApp.git
    cd Image_Classification_WebApp
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

5. **Open the app in your browser at `http://localhost:8501`.**

## Dependencies

The app requires the following Python packages:

- `streamlit`
- `tensorflow`
- `Pillow`
- `numpy`
- `matplotlib`

