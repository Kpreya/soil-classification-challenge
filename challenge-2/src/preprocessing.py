import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageProcessor:
    """Handle various image quality issues"""
    
    @staticmethod
    def assess_image_quality(img_path):
        """Assess image quality and return quality score (0-1)"""
        try:
            # Load with OpenCV for analysis
            cv_img = cv2.imread(str(img_path))
            if cv_img is None:
                return 0.0
            
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Check for blur (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_normalized = min(blur_score / 500, 1.0)  # Normalize
            
            # Check brightness
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Penalize very dark/bright
            
            # Check contrast
            contrast = gray.std() / 128.0
            contrast_score = min(contrast, 1.0)
            
            # Combined quality score
            quality = (blur_normalized * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
            return quality
            
        except Exception:
            return 0.0
    
    @staticmethod
    def enhance_image(pil_img):
        """Enhance image quality - reduce noise, improve contrast"""
        try:
            # Convert to numpy for OpenCV operations
            img_np = np.array(pil_img)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
            
            # Convert back to PIL
            enhanced = Image.fromarray(denoised)
            
            # Enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            return enhanced
        except Exception:
            return pil_img


class NegativeGenerator:
    """Generate non-soil images from soil images"""
    
    def __init__(self):
        self.transformations = [
            self._create_abstract_pattern,
            self._create_heavy_blur,
            self._create_noise_pattern,
            self._create_geometric_shapes,
            self._create_color_distortion,
            self._create_texture_removal
        ]
    
    def _create_abstract_pattern(self, img):
        """Create abstract patterns that don't look like soil"""
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Create wave patterns
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        # Create wave interference pattern
        wave1 = np.sin(X * 0.1) * np.cos(Y * 0.1)
        wave2 = np.cos(X * 0.05) * np.sin(Y * 0.15)
        pattern = ((wave1 + wave2) * 127 + 128).astype(np.uint8)
        
        # Apply to all channels
        for c in range(3):
            img_np[:, :, c] = pattern
            
        return Image.fromarray(img_np)
    
    def _create_heavy_blur(self, img):
        """Apply heavy blur to destroy soil texture"""
        # Multiple blur applications
        blurred = img.filter(ImageFilter.GaussianBlur(radius=20))
        blurred = blurred.filter(ImageFilter.GaussianBlur(radius=15))
        return blurred
    
    def _create_noise_pattern(self, img):
        """Add heavy noise to mask soil features"""
        img_np = np.array(img)
        
        # Add random noise
        noise = np.random.randint(0, 100, img_np.shape, dtype=np.uint8)
        noisy = cv2.addWeighted(img_np, 0.3, noise, 0.7, 0)
        
        return Image.fromarray(noisy)
    
    def _create_geometric_shapes(self, img):
        """Create artificial geometric patterns"""
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Create checkerboard pattern
        block_size = 30
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if (i//block_size + j//block_size) % 2 == 0:
                    end_i = min(i + block_size, h)
                    end_j = min(j + block_size, w)
                    img_np[i:end_i, j:end_j] = [150, 150, 150]
                else:
                    end_i = min(i + block_size, h)
                    end_j = min(j + block_size, w)
                    img_np[i:end_i, j:end_j] = [50, 50, 50]
                    
        return Image.fromarray(img_np)
    
    def _create_color_distortion(self, img):
        """Apply extreme color distortion"""
        img_np = np.array(img)
        
        # Extreme color channel manipulation
        img_np[:,:,0] = np.clip(img_np[:,:,0] * 2.5, 0, 255)  # Red boost
        img_np[:,:,1] = img_np[:,:,1] // 4  # Green reduction
        img_np[:,:,2] = np.clip(img_np[:,:,2] * 1.8, 0, 255)  # Blue boost
        
        # Add color gradient
        h, w = img_np.shape[:2]
        gradient = np.linspace(0, 100, w)
        for i in range(h):
            img_np[i, :, 1] = np.clip(img_np[i, :, 1] + gradient, 0, 255)
            
        return Image.fromarray(img_np.astype(np.uint8))
    
    def _create_texture_removal(self, img):
        """Remove natural textures and add artificial ones"""
        img_np = np.array(img)
        
        # Apply strong median filter to remove texture
        for channel in range(img_np.shape[2]):
            img_np[:,:,channel] = cv2.medianBlur(img_np[:,:,channel], 21)
        
        # Add artificial linear patterns
        h, w = img_np.shape[:2]
        for i in range(0, h, 5):
            img_np[i:i+2, :] = [200, 200, 200]
            
        return Image.fromarray(img_np.astype(np.uint8))
    
    def generate_negative(self, soil_img):
        """Generate a non-soil image"""
        transform = random.choice(self.transformations)
        return transform(soil_img)


class SoilDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False, 
                 add_negatives=False, negative_ratio=0.4, enhance_images=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.enhance_images = enhance_images
        self.processor = ImageProcessor()
        self.neg_generator = NegativeGenerator() if add_negatives else None
        
        # Add synthetic negatives if requested
        if add_negatives and not is_test:
            self._add_synthetic_negatives(negative_ratio)
        
        # Assess image quality
        if not is_test:
            self._assess_quality()

    def _add_synthetic_negatives(self, ratio):
        """Add synthetic negative samples"""
        n_negatives = int(len(self.df) * ratio)
        
        # Sample images to create negatives from
        negative_indices = np.random.choice(len(self.df), n_negatives, replace=True)
        negative_samples = self.df.iloc[negative_indices].copy()
        negative_samples['label'] = 0  # Set as negative class
        negative_samples['is_synthetic'] = True
        
        # Add synthetic flag to original data
        self.df['is_synthetic'] = False
        
        # Combine datasets
        self.df = pd.concat([self.df, negative_samples], ignore_index=True)
        
        print(f"Added {n_negatives} synthetic negative samples")
        print(f"Total samples: {len(self.df)} (Positive: {sum(self.df['label'] == 1)}, Negative: {sum(self.df['label'] == 0)})")

    def _assess_quality(self):
        """Assess quality of all images"""
        print("Assessing image quality...")
        quality_scores = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Quality check"):
            if row.get('is_synthetic', False):
                quality_scores.append(1.0)  # Synthetic images are "good quality"
            else:
                img_path = os.path.join(self.img_dir, row["image_id"])
                score = self.processor.assess_image_quality(img_path)
                quality_scores.append(score)
        
        self.df['quality_score'] = quality_scores
        
        # Report quality statistics
        avg_quality = np.mean(quality_scores)
        low_quality_count = sum(s < 0.3 for s in quality_scores)
        print(f"Average quality score: {avg_quality:.3f}")
        print(f"Low quality images (<0.3): {low_quality_count}/{len(quality_scores)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Generate synthetic negative if needed
            if not self.is_test and row.get('is_synthetic', False) and row['label'] == 0:
                img = self.neg_generator.generate_negative(img)
            else:
                # Enhance real images if quality is low
                if (not self.is_test and 
                    self.enhance_images and 
                    row.get('quality_score', 1.0) < 0.4):
                    img = self.processor.enhance_image(img)
                    
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Create a default image
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            img = self.transform(img)
            
        if self.is_test:
            return img, 0
            
        label = int(row["label"])
        return img, label


def get_transforms():
    """Get train and validation transforms"""
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    return train_tf, val_tf