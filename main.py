import streamlit as st
import base64
import pandas as pd
import matplotlib.pyplot as plt
import cv2






csv_file_path = "counts_vid_seconds.csv"
df = pd.read_csv(csv_file_path)
x = 4

# Add a custom sidebar layout with logo and colored border
st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Function to convert image to base64
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load and display sidebar image with glowing effect
img_path = "smartislogo.png"
img_base64 = img_to_base64(img_path)
st.sidebar.markdown(
    f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# Sidebar for Mode Selection
mode = st.sidebar.radio("Select Mode:", options=["Live detection", "Single angle", "Multiple angles", "Data insight"], index=0)
st.sidebar.markdown("---")

# pretrained_model = cv2.CascadeClassifier(r"face_detector.xml")


if mode == "Live detection":
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # coordinate_list = pretrained_model.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)

        person_count = 0  # Reset person count for each frame

        # Drawing rectangle in frame and updating person count
        # for (x, y, w, h) in coordinate_list:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #     person_count += 1

        # Display detected face, person count, and wait time
        cv2.putText(frame, f'Person Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(frame, 'Press "q" to quit', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')
    

if mode == "Single angle":
    st.title("Object Detection and Tracking - Single angle")
    vid1 = "output_video.mp4"
    st.video(vid1)

    

        

elif mode == "Multiple angles":
    st.title("Object Detection and Tracking - Multiple angles")
    vid2 = "hall1hall2.mp4"
    st.video(vid2)

elif mode == "Data insight":
    st.title("Object Detection and Tracking - Data Insight")
    vid3 = "output_video.mp4"
    st.video(vid3)


    fig, ax1 = plt.subplots()
    ax1.plot(df["Time (seconds).1"], df["Player Count"], color="blue")
    ax1.set_xlabel("Time (seconds).1")
    ax1.set_ylabel("Player Count")
    ax1.set_title("Player Count vs Time")
    ax1.set_yticks(range(0,15))

    st.pyplot(fig)

    # Plot 2: Line plot of Bystander Count vs Time
    fig, ax2 = plt.subplots()
    ax2.plot(df["Time (seconds).1"], df["Bystander Count"], color="green")
    ax2.set_xlabel("Time (seconds).1")
    ax2.set_ylabel("Bystander Count")
    ax2.set_title("Bystander Count vs Time")
    ax2.set_yticks(range(0, 15))

    st.pyplot(fig)

    # Plot 3: Bar graph of Minimum and Maximum Player Count and Bystander Count
    min_max_data = {
        "Category": ["Minimum", "Maximum"],
        "Player Count": [df["Player Count"].min(), df["Player Count"].max()],
        "Bystander Count": [df["Bystander Count"].min(), df["Bystander Count"].max()]
    }

    min_max_df = pd.DataFrame(min_max_data)

    fig, ax3 = plt.subplots()
    min_max_df.plot(kind="bar", x="Category", ax=ax3)
    ax3.set_ylabel("Count")
    ax3.set_title("Minimum and Maximum Player Count and Bystander Count")

    st.pyplot(fig)