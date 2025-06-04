import streamlit as st
import cv2
import numpy as np
import os

# Define the input and output directories
input_folder = r'D:\mp6\images\moon_imgs'
output_folder = r'D:\mp6\images\processed_imgs'

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to detect pits and boulders in an image
def detect_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

    # HoughCircles for detecting pits (circular shapes)
    circles = cv2.HoughCircles(
        blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=100, param2=30, minRadius=10, maxRadius=100
    )

    # Detect contours for boulders
    _, threshold_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()  # Keep the original image intact

    # Mark the detected pits (craters)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw green circle around the pit
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)

    # Mark the detected boulders
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            # Draw green rectangle around the boulder
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output_image

# Dashboard navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Moon Path Navigator"])

# Home Page
if app_mode == "Home":
    st.header("Moon Path Navigator")
    st.image("home.jpg", use_column_width=True)
    st.markdown("""
    The Moon Path Navigator project aims to chart a safe, scientifically valuable route for a lunar rover on the Moon's south pole using high-resolution Chandrayaan-2 data. The path avoids obstacles like craters and boulders and focuses on high-interest sites such as potential water ice deposits and unique geological features.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### Objective
    - Analyse the landing site region in the south polar region of the Moon.
    - Detect and annotate major features like craters and boulders in lunar images.

    #### Expected Outcome
    - Annotated maps/images with clear markings of craters and boulders.
    """)

# Moon Path Navigator Page
elif app_mode == "Moon Path Navigator":
    st.header("Moon Path Navigator")
    test_image = st.file_uploader("Choose a Moon Image:")

    if test_image and st.button("Process Image"):
        # Load the image
        image = cv2.imdecode(np.frombuffer(test_image.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            # Process the image to detect features
            processed_image = detect_features(image)

            # Display the processed image
            st.image(processed_image, channels="BGR", use_column_width=True)
            st.success("Craters and boulders detected and marked.")
        else:
            st.error("Error loading the image. Please ensure the file is a valid image.")

    st.write("Upload a moon image to detect and mark craters and boulders.")
