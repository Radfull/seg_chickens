import cv2
import numpy as np
from pathlib import Path

def augment_image(image):

    rows, cols = image.shape[:2]
    
    aug_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(aug_img)
    
    h += np.random.randint(0, 100, size=(rows, cols), dtype=np.uint8)
    s += np.random.randint(0, 20, size=(rows, cols), dtype=np.uint8)
    v += np.random.randint(0, 10, size=(rows, cols), dtype=np.uint8)
    
    aug_img = cv2.merge([h, s, v])
    aug_img = cv2.cvtColor(aug_img, cv2.COLOR_HSV2RGB)
    
    blur_val = np.random.randint(5, 12)
    aug_img = cv2.blur(aug_img, (blur_val, blur_val))
    
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.randint(-30, 30)
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
    
    noise = np.random.normal(0, 0.2, image.shape).astype(np.uint8)
    aug_img = cv2.add(aug_img, noise)
    
    return aug_img

def augment_dataset(images_dir, labels_dir, output_dir, num_augmentations=1):

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    aug_images_dir = output_dir / 'images'
    aug_labels_dir = output_dir / 'labels'
    aug_images_dir.mkdir(parents=True, exist_ok=True)
    aug_labels_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(images_dir.glob("*.jpg"))
    
    print(f"{len(image_files)} images find")
    
    for i, image_path in enumerate(image_files):
        print(f"processing {i+1}/{len(image_files)}: {image_path.name}")
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        output_image_path = aug_images_dir / image_path.name
        cv2.imwrite(str(output_image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        if label_path.exists():
            output_label_path = aug_labels_dir / f"{image_path.stem}.txt"
            output_label_path.write_text(label_path.read_text())
        
        for aug_idx in range(num_augmentations):
            aug_image = augment_image(image)
            
            aug_image_name = f"{image_path.stem}_aug_{aug_idx+1}{image_path.suffix}"
            aug_image_path = aug_images_dir / aug_image_name
            cv2.imwrite(str(aug_image_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            
            if label_path.exists():
                aug_label_name = f"{image_path.stem}_aug_{aug_idx+1}.txt"
                aug_label_path = aug_labels_dir / aug_label_name
                aug_label_path.write_text(label_path.read_text())
    
    print("Augmentation is complete")

if __name__ == "__main__":
    images_dir = Path('dataset/val/images')
    labels_dir = Path('dataset/val/labels')
    output_dir = Path('augmented_val')
    
    augment_dataset(images_dir, labels_dir, output_dir, num_augmentations=1)