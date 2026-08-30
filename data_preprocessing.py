import cv2
import random
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


RAW_DATA_DIR = "./Celebrity Faces Dataset"          
OUTPUT_DIR = "dataset_split"                        
IMG_TARGET_SIZE = (224, 224)                        
CONFIDENCE_THRESHOLD = 0.5                          
TRAIN_RATIO = 0.8                                   
RANDOM_SEED = 42                                    

model = YOLO("yolov8n.pt") 

def process_dataset(raw_dir, output_dir, target_size):
    random.seed(RANDOM_SEED) 
    raw_path = Path(raw_dir)
    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    celebrity_folders = [f for f in raw_path.iterdir() if f.is_dir()]

    for celeb_folder in celebrity_folders:
        celeb_name = celeb_folder.name
        
        image_files = [f for f in celeb_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        random.shuffle(image_files)
        
        split_idx = int(len(image_files) * TRAIN_RATIO)
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]
        
        (train_dir / celeb_name).mkdir(parents=True, exist_ok=True)
        (test_dir / celeb_name).mkdir(parents=True, exist_ok=True)


        def crop_and_save(files_list, save_base_dir, desc_label, is_train=False):
            for img_file in tqdm(files_list, desc=f"处理 {celeb_name} ({desc_label})"):
                img = cv2.imread(str(img_file))
                if img is None: continue

                h, w, _ = img.shape
                results = model.predict(source=img, conf=CONFIDENCE_THRESHOLD, verbose=False)
                
                best_box = None
                max_conf = 0.0

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        if int(box.cls[0]) == 0:
                            conf = float(box.conf[0])
                            if conf > max_conf:
                                max_conf = conf
                                best_box = box.xyxy[0].cpu().numpy()

                if best_box is not None:
                    xmin, ymin, xmax, ymax = map(int, best_box)
                    pad_x = int((xmax - xmin) * 0.08)
                    pad_y = int((ymax - ymin) * 0.08)
                    x1, y1 = max(0, xmin - pad_x), max(0, ymin - pad_y)
                    x2, y2 = min(w, xmax + pad_x), min(h, ymax + pad_y)
                    cropped_face = img[y1:y2, x1:x2]
                else:
                    cropped_face = img

                if cropped_face.size != 0:
                    resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
                    

                    output_file = save_base_dir / celeb_name / img_file.name
                    cv2.imwrite(str(output_file), resized_face)

                    if is_train:
                        # rotate
                        flipped_face = cv2.flip(resized_face, 1)
                        flip_file = save_base_dir / celeb_name / f"{img_file.stem}_flip{img_file.suffix}"
                        cv2.imwrite(str(flip_file), flipped_face)
                        
                        # bright
                        bright_face = cv2.convertScaleAbs(resized_face, alpha=1.2, beta=20)
                        bright_file = save_base_dir / celeb_name / f"{img_file.stem}_bright{img_file.suffix}"
                        cv2.imwrite(str(bright_file), bright_face)

        
        crop_and_save(train_files, train_dir, "Train", is_train=True)
        crop_and_save(test_files, test_dir, "Test", is_train=False)
        

if __name__ == "__main__":
    process_dataset(RAW_DATA_DIR, OUTPUT_DIR, IMG_TARGET_SIZE)