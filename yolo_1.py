import cv2
from ultralytics import YOLO
def naach_chamiya():
    in_pass = 'image.png'
    out_pass = 'image.png'
    chamiya_ka_naam = 'yolov8n.pt'  
    print(f"Loading YOLO model: {chamiya_ka_naam}...")
    try:
        model = YOLO(chamiya_ka_naam)
        print(f"Attempting to read image from: {in_pass}")
        image = cv2.imread(in_pass)
        if image is None:
            print(f"ERROR: Could not load image at path: {in_pass}")
            print("Please ensure the image file exists and the path is correct.")
            return
        print("Running inference...")
        results = model(image, conf=0.25)
        result = results[0]
        annotated_image = result.plot()
        cv2.imwrite(out_pass, annotated_image)
        print(f"Detection successful! Result saved to {out_pass}")
        cv2.imshow("YOLOv8 Detection Result", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Display window closed.")
    except Exception as e:
        print(f"\nerror aala:")
        print(f"Error details: {e}")
        print("plz run pip install ultralytics opencv-python")
if __name__ == "__main__":
    naach_chamiya()