import cv2
from ultralytics import YOLO
def naach_chamiya():
    in_pass = 'image.png'
    out_pass = 'result_image.png'
    chamiya_ka_naam = 'yolov8n.pt'  
    print(f"loading yolov8{chamiya_ka_naam}...")
    try:
        model = YOLO(chamiya_ka_naam)
        print(f"img_read started{in_pass}")
        image = cv2.imread(in_pass)
        if image is None:
            print(f"error img_path:{in_pass}")
            print("img input error")
            return
        print("Running inference...")
        results = model(image, conf=0.25)
        result = results[0]
        annotated_image = result.plot()
        cv2.imwrite(out_pass, annotated_image)
        print(f"detection successful_Result saved to {out_pass}")
        cv2.imshow("result of yolov8 out", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("dhwar baand")
    except Exception as e:
        print(f"\nerror aala:")
        print(f"Error details: {e}")
        print("plz run pip install ultralytics opencv-python")
if __name__ == "__main__":
    naach_chamiya()