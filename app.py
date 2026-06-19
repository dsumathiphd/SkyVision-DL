import os
import cv2
import json
import base64
import tempfile
import requests
import pandas as pd
import streamlit as st

# ---------- Page Config ----------
st.set_page_config(page_title="SkyVision V1", layout="wide")
st.title("SkyVision V1 - Facade Glass Detection System")

# ---------- API Credentials (hard‑coded for college project) ----------
API_KEY = "qieHSP5FdT7g4Exn3U9g"
MODEL_ID = "door-window-cwazm/2"

# ---------- Operating Mode ----------
mode = st.radio("Select Operating Mode", ["Telemetry Mode", "Reference-Based Mode"])

# ---------- File Upload ----------
uploaded_file = st.file_uploader("Upload a building video", type=["mp4", "avi", "mov", "mpeg4"])
telemetry_file = None
if mode == "Telemetry Mode":
    telemetry_file = st.file_uploader("Upload telemetry CSV", type=["csv"])

# ---------- Reference Inputs ----------
if mode == "Reference-Based Mode":
    st.subheader("Reference Inputs")
    col1, col2 = st.columns(2)
    with col1:
        first_floor_height = st.number_input("First floor height (m)", min_value=0.1, value=3.5)
        floor_height = st.number_input("Each floor height (m)", min_value=0.1, value=3.0)
    with col2:
        glass_height = st.number_input("Average glass height (m)", min_value=0.1, value=1.5)
        total_building_height = st.number_input("Total building height (m)", min_value=0.1, value=30.0)

# ---------- Helper Functions ----------
def seconds_to_hms(seconds: float) -> str:
    seconds = int(seconds)
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hrs:02}:{mins:02}:{secs:02}"

def get_facade_zone(y_center: float, frame_height: int) -> str:
    if y_center < frame_height / 3:
        return "Upper Section"
    elif y_center < 2 * frame_height / 3:
        return "Mid Section"
    else:
        return "Lower Section"

def get_nearest_telemetry(timestamp_sec: float, telemetry_df: pd.DataFrame) -> dict | None:
    """Find the closest telemetry row without mutating the original dataframe."""
    if telemetry_df is None or telemetry_df.empty:
        return None
    time_diff = (telemetry_df["timestamp"] - timestamp_sec).abs()
    idx = time_diff.idxmin()
    row = telemetry_df.loc[idx]
    return {
        "latitude": row.get("latitude", None),
        "longitude": row.get("longitude", None),
        "altitude": row.get("altitude", None)
    }

def estimate_height_reference(y_center: float, frame_height: int, total_height_m: float) -> float:
    """Estimate height assuming the top of the frame aligns with the building top."""
    relative_from_top = y_center / frame_height
    estimated_height = total_height_m * (1 - relative_from_top)
    return round(estimated_height, 2)

def frame_to_base64(frame: cv2.Mat) -> str:
    """Encode an OpenCV frame to a base64 JPEG string (no disk write)."""
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        raise ValueError("Failed to encode frame")
    return base64.b64encode(buffer).decode("utf-8")

# ---------- Main Processing ----------
if uploaded_file is not None:
    # Load telemetry if provided
    telemetry_df = None
    if mode == "Telemetry Mode" and telemetry_file is not None:
        try:
            telemetry_df = pd.read_csv(telemetry_file)
            # Ensure timestamp column is numeric
            telemetry_df["timestamp"] = pd.to_numeric(telemetry_df["timestamp"], errors="coerce")
        except Exception as e:
            st.error(f"Error reading telemetry CSV: {e}")
            st.stop()

    # Write uploaded video to a temporary file (OpenCV needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    st.subheader("Uploaded Video")
    st.video(video_path)

    # Prepare output
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)
    FRAME_SKIP = 30

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1  # avoid division by zero

    frame_index = 0
    processed_frames = []
    detections_output = []
    detection_counter = 1

    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress
        progress = min(frame_index / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_index}/{total_frames}")

        if frame_index % FRAME_SKIP != 0:
            frame_index += 1
            continue

        # Encode frame directly to base64 (no temp file)
        try:
            image_b64 = frame_to_base64(frame)
        except Exception:
            frame_index += 1
            continue

        # Call Roboflow API
        url = f"https://detect.roboflow.com/{MODEL_ID}?api_key={API_KEY}"
        try:
            response = requests.post(
                url,
                data=image_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            st.warning(f"API request failed at frame {frame_index}: {e}")
            frame_index += 1
            continue

        predictions = result.get("predictions", [])
        frame_height, frame_width = frame.shape[:2]
        timestamp_sec = frame_index / fps
        timestamp_str = seconds_to_hms(timestamp_sec)

        # Annotate frame only if there is at least one valid detection
        has_valid_detection = False
        annotated_frame = frame.copy()

        for pred in predictions:
            if pred["confidence"] < 0.4:
                continue
            has_valid_detection = True

            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]
            label = pred["class"]
            confidence = pred["confidence"]

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"{label} {confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            facade_zone = get_facade_zone(y, frame_height)

            latitude = longitude = altitude = estimated_height = None

            if mode == "Telemetry Mode" and telemetry_df is not None:
                telemetry_data = get_nearest_telemetry(timestamp_sec, telemetry_df)
                if telemetry_data:
                    latitude = telemetry_data["latitude"]
                    longitude = telemetry_data["longitude"]
                    altitude = telemetry_data["altitude"]
                    # Simple approximation: use altitude directly (to be replaced by a proper height model)
                    estimated_height = altitude
            else:
                estimated_height = estimate_height_reference(y, frame_height, total_building_height)

            detections_output.append({
                "detection_id": f"{detection_counter:03}",
                "class": label,
                "confidence": round(confidence, 2),
                "frame": frame_index,
                "timestamp": timestamp_str,
                "latitude": latitude,
                "longitude": longitude,
                "facade_zone": facade_zone,
                "estimated_height_m": estimated_height
            })
            detection_counter += 1

        if has_valid_detection:
            output_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
            cv2.imwrite(output_path, annotated_frame)
            processed_frames.append(output_path)

        frame_index += 1

    cap.release()
    # Clean up temporary video file
    try:
        os.unlink(video_path)
    except Exception:
        pass

    progress_bar.empty()
    status_text.empty()

    st.success(f"Processed {len(processed_frames)} frames with detections")
    st.write(f"Total detections found: {len(detections_output)}")

    # Display processed frames (limit to first 20 for performance)
    st.subheader("Analyzed Frames")
    max_display = min(20, len(processed_frames))
    for i, img_path in enumerate(processed_frames[:max_display]):
        st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    if len(processed_frames) > max_display:
        st.info(f"Showing {max_display} of {len(processed_frames)} frames. Download JSON for full details.")

    # Show detection records as a table instead of raw JSON per record
    st.subheader("Detection Records")
    if detections_output:
        df_out = pd.DataFrame(detections_output)
        st.dataframe(df_out, use_container_width=True)

        # JSON download
        json_data = json.dumps(detections_output, indent=4)
        st.download_button(
            label="Download Detections JSON",
            data=json_data,
            file_name="detections.json",
            mime="application/json"
        )

        # CSV download
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Detections CSV",
            data=csv,
            file_name="detections.csv",
            mime="text/csv"
        )
    else:
        st.info("No detections met the confidence threshold.")
else:
    st.info("Please upload a video file to begin.")