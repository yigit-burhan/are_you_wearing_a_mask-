"""
Automated script to download and setup the Face Mask 12K Images Dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset
"""

import os
import shutil
from pathlib import Path

# ============================================================================
# STEP 1: Download Dataset from Kaggle
# ============================================================================

def download_dataset():
    """
    Download the Face Mask 12K dataset from Kaggle
    
    Prerequisites:
    1. pip install kaggle
    2. Set up Kaggle API credentials (kaggle.json)
    """
    print("=" * 70)
    print("DOWNLOADING FACE MASK 12K DATASET FROM KAGGLE")
    print("=" * 70)
    
    try:
        import kaggle
        
        print("\n✓ Kaggle library found")
        print("Downloading dataset (this may take a few minutes)...")
        
        # Download the specific dataset
        kaggle.api.dataset_download_files(
            'ashishjangra27/face-mask-12k-images-dataset',
            path='.',
            unzip=True
        )
        
        print("✓ Dataset downloaded and extracted successfully!")
        return True
        
    except ImportError:
        print("❌ Kaggle library not installed")
        print("\nPlease run: pip install kaggle")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have set up kaggle.json")
        print("2. Go to: https://www.kaggle.com/settings")
        print("3. Click 'Create New API Token'")
        print("4. Place kaggle.json in:")
        print("   - Windows: C:\\Users\\<username>\\.kaggle\\")
        print("   - Mac/Linux: ~/.kaggle/")
        return False


# ============================================================================
# STEP 2: Organize Dataset Structure
# ============================================================================

def organize_12k_dataset():
    """
    Organize the downloaded dataset into the required structure
    
    Expected structure after download:
    Face Mask Dataset/
    ├── Train/
    │   ├── WithMask/
    │   └── WithoutMask/
    ├── Validation/
    │   ├── WithMask/
    │   └── WithoutMask/
    └── Test/
        ├── WithMask/
        └── WithoutMask/
    
    Target structure:
    mask_dataset/
    ├── with_mask/
    └── without_mask/
    """
    
    print("\n" + "=" * 70)
    print("ORGANIZING DATASET")
    print("=" * 70)
    
    # Find the extracted folder (common names)
    possible_names = [
        'Face Mask Dataset',
        'face-mask-12k-images-dataset',
        'archive',
        'Face-Mask-Dataset'
    ]
    
    source_dir = None
    for name in possible_names:
        if os.path.exists(name):
            source_dir = name
            print(f"\n✓ Found dataset folder: {name}")
            break
    
    if not source_dir:
        print("❌ Could not find extracted dataset folder")
        print("Please check if the download was successful")
        return False
    
    # Create target directories
    target_dir = 'mask_dataset'
    os.makedirs(f"{target_dir}/with_mask", exist_ok=True)
    os.makedirs(f"{target_dir}/without_mask", exist_ok=True)
    
    print(f"\nOrganizing images into {target_dir}/...")
    
    # Copy from Train, Validation, and Test folders
    splits = ['Train', 'Validation', 'Test']
    total_with_mask = 0
    total_without_mask = 0
    
    for split in splits:
        split_path = Path(source_dir) / split
        
        if not split_path.exists():
            print(f"⚠ Warning: {split} folder not found, skipping...")
            continue
        
        # Copy WithMask images
        with_mask_src = split_path / 'WithMask'
        if with_mask_src.exists():
            count = copy_images(with_mask_src, f"{target_dir}/with_mask", split)
            total_with_mask += count
            print(f"  {split} - With Mask: {count} images")
        
        # Copy WithoutMask images
        without_mask_src = split_path / 'WithoutMask'
        if without_mask_src.exists():
            count = copy_images(without_mask_src, f"{target_dir}/without_mask", split)
            total_without_mask += count
            print(f"  {split} - Without Mask: {count} images")
    
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"✓ With Mask: {total_with_mask} images")
    print(f"✓ Without Mask: {total_without_mask} images")
    print(f"✓ Total: {total_with_mask + total_without_mask} images")
    print(f"\n✓ Dataset ready at: {target_dir}/")
    print("=" * 70)
    
    return True


def copy_images(source_dir, target_dir, prefix=''):
    """
    Copy images from source to target directory with unique naming
    
    Args:
        source_dir: Source directory path
        target_dir: Target directory path
        prefix: Prefix for filename (e.g., 'Train', 'Test')
    
    Returns:
        Number of images copied
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    count = 0
    for ext in image_extensions:
        for img_path in source_dir.glob(ext):
            # Create unique filename with prefix
            new_name = f"{prefix}_{img_path.name}" if prefix else img_path.name
            target_path = target_dir / new_name
            
            # Handle duplicates
            counter = 1
            while target_path.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                new_name = f"{prefix}_{stem}_{counter}{suffix}"
                target_path = target_dir / new_name
                counter += 1
            
            shutil.copy2(img_path, target_path)
            count += 1
    
    return count


# ============================================================================
# STEP 3: Verify Dataset
# ============================================================================

def verify_dataset(dataset_dir='mask_dataset'):
    """Verify the organized dataset"""
    
    print("\n" + "=" * 70)
    print("VERIFYING DATASET")
    print("=" * 70)
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    with_mask_dir = Path(dataset_dir) / 'with_mask'
    without_mask_dir = Path(dataset_dir) / 'without_mask'
    
    # Count images
    with_mask_count = len(list(with_mask_dir.glob('*.jpg'))) + \
                     len(list(with_mask_dir.glob('*.png')))
    without_mask_count = len(list(without_mask_dir.glob('*.jpg'))) + \
                        len(list(without_mask_dir.glob('*.png')))
    
    print(f"\n✓ With Mask directory: {with_mask_count} images")
    print(f"✓ Without Mask directory: {without_mask_count} images")
    
    if with_mask_count == 0 or without_mask_count == 0:
        print("\n⚠ Warning: One or both categories have no images!")
        return False
    
    # Check balance
    ratio = min(with_mask_count, without_mask_count) / max(with_mask_count, without_mask_count)
    print(f"\nClass balance ratio: {ratio:.2f}")
    
    if ratio < 0.5:
        print("⚠ Warning: Dataset is imbalanced (one class has less than 50% of the other)")
    else:
        print("✓ Dataset is reasonably balanced")
    
    # Show sample images
    print("\nSample images:")
    with_mask_samples = list(with_mask_dir.glob('*.jpg'))[:3]
    without_mask_samples = list(without_mask_dir.glob('*.jpg'))[:3]
    
    print("\nWith Mask:")
    for img in with_mask_samples:
        print(f"  - {img.name}")
    
    print("\nWithout Mask:")
    for img in without_mask_samples:
        print(f"  - {img.name}")
    
    print("\n" + "=" * 70)
    print("✓ DATASET VERIFICATION COMPLETE")
    print("=" * 70)
    
    return True


# ============================================================================
# STEP 4: Setup Kaggle API (Helper)
# ============================================================================

def setup_kaggle_credentials():
    """Interactive setup for Kaggle API credentials"""
    
    print("\n" + "=" * 70)
    print("KAGGLE API CREDENTIALS SETUP")
    print("=" * 70)
    
    import platform
    system = platform.system()
    
    if system == "Windows":
        kaggle_dir = os.path.expanduser("~\\.kaggle")
    else:
        kaggle_dir = os.path.expanduser("~/.kaggle")
    
    print("\nSteps to set up Kaggle API:")
    print("\n1. Go to: https://www.kaggle.com/settings")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This will download 'kaggle.json'")
    print(f"\n5. Move kaggle.json to: {kaggle_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(kaggle_dir, exist_ok=True)
    print(f"\n✓ Created directory: {kaggle_dir}")
    
    if system != "Windows":
        kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
        print(f"\n6. After moving the file, run this command:")
        print(f"   chmod 600 {kaggle_json_path}")
    
    print("\n" + "=" * 70)
    print("After setup, run this script again to download the dataset")
    print("=" * 70)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "=" * 70)
    print("FACE MASK 12K DATASET IMPORTER")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Download the Face Mask 12K dataset from Kaggle")
    print("2. Organize it into the required structure")
    print("3. Verify the dataset is ready for training")
    
    # Check if dataset already exists
    if os.path.exists('mask_dataset'):
        print("\n⚠ Dataset directory 'mask_dataset' already exists")
        choice = input("Do you want to re-download and overwrite? (yes/no): ").strip().lower()
        if choice not in ['yes', 'y']:
            print("Exiting without changes.")
            return
        else:
            print("Removing existing dataset...")
            shutil.rmtree('mask_dataset')
    
    # Try to download
    print("\nAttempting to download dataset...")
    success = download_dataset()
    
    if not success:
        print("\n" + "=" * 70)
        print("SETUP REQUIRED")
        print("=" * 70)
        setup_kaggle_credentials()
        return
    
    # Organize dataset
    print("\nDataset downloaded successfully!")
    success = organize_12k_dataset()
    
    if not success:
        print("\n❌ Failed to organize dataset")
        print("Please check the extracted folder structure")
        return
    
    # Verify dataset
    verify_dataset()
    
    print("\n" + "=" * 70)
    print("✅ ALL DONE!")
    print("=" * 70)
    print("\nYour dataset is ready for training!")
    print("Next step: Run your training script")
    print("  python mask_detector.py")
    print("=" * 70)


# ============================================================================
# ALTERNATIVE: Manual organization if already downloaded
# ============================================================================

def organize_manual(extracted_folder):
    """
    Organize manually downloaded dataset
    
    Usage:
        organize_manual('Face Mask Dataset')
    """
    if not os.path.exists(extracted_folder):
        print(f"❌ Folder not found: {extracted_folder}")
        return False
    
    # Temporarily set as source_dir for organize function
    original_exists = {}
    for name in ['Face Mask Dataset', 'face-mask-12k-images-dataset', 'archive']:
        if os.path.exists(name):
            original_exists[name] = True
    
    # Create symlink or rename temporarily
    temp_name = 'Face Mask Dataset'
    if extracted_folder != temp_name and not os.path.exists(temp_name):
        os.rename(extracted_folder, temp_name)
        organize_12k_dataset()
        if not os.path.exists(extracted_folder):
            os.rename(temp_name, extracted_folder)
    else:
        organize_12k_dataset()
    
    verify_dataset()


# ============================================================================
# RUN SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()