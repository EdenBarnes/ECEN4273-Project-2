import cv2
from ultralytics import YOLO

def process_video(weights_path: str, source: str | int, output_path: str | None,
                  conf: float = 0.25, live_stream: bool = False):
    """Run object detection on an .mp4 file or live camera stream."""
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(source)  # Path to .mp4, or 0 for webcam

    writer = None
    if not live_stream and output_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()  # Draws boxes, labels, and confidences

        if live_stream:
            cv2.imshow("Object Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif writer:
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()