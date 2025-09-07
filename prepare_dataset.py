import shutil
import random
from pathlib import Path
import yaml

def create_struct():
    
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    return dataset_dir

def split_dataset(data_dir, dataset_dir, train_ratio=0.8, val_ratio=0.1):
    
    images_dir = Path(data_dir) / "images"
    labels_dir = Path(data_dir) / "labels"
    
    image_files = list(images_dir.glob("*.jpg"))
    
    random.shuffle(image_files)
    
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"Total: {total_files}")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")
    
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for img_file in files:

            dst_img = dataset_dir / split / 'images' / img_file.name
            shutil.copy2(img_file, dst_img)
            
            label_file = labels_dir / (img_file.stem + '.txt')
            if label_file.exists():
                dst_label = dataset_dir / split / 'labels' / label_file.name
                shutil.copy2(label_file, dst_label)
            else:
                print(f"not find: {img_file.name}")

def create_yaml_config(dataset_dir, num_classes=1):
    
    config = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': num_classes,  
        'names': ['chicken']
    }
    
    config_path = dataset_dir / 'data.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config_path

def main():
    
    random.seed(42)
    dataset_dir = create_struct()
    split_dataset("data", dataset_dir)
    create_yaml_config(dataset_dir)
    
if __name__ == "__main__":
    main()
