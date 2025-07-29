import numpy as np
import torch
from PIL import Image
from typing import Optional, Dict, Any


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    Applies frequency and time masking to spectrograms
    """
    
    def __init__(
        self, 
        freq_mask_param: int = 20, 
        time_mask_param: int = 40, 
        num_freq_masks: int = 1, 
        num_time_masks: int = 1, 
        mask_value: float = 0.0
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        tensor = tensor.clone()
        _, freq_dim, time_dim = tensor.shape
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            freq_mask_width = np.random.randint(1, min(self.freq_mask_param, freq_dim) + 1)
            freq_start = np.random.randint(0, freq_dim - freq_mask_width + 1)
            tensor[:, freq_start:freq_start + freq_mask_width, :] = self.mask_value
        
        # Time masking
        for _ in range(self.num_time_masks):
            time_mask_width = np.random.randint(1, min(self.time_mask_param, time_dim) + 1)
            time_start = np.random.randint(0, time_dim - time_mask_width + 1)
            tensor[:, :, time_start:time_start + time_mask_width] = self.mask_value
        
        return tensor


class SpectrogramNoise:
    """
    Adds controlled Gaussian noise to spectrograms
    """
    
    def __init__(self, noise_factor: float = 0.01, probability: float = 0.5):
        self.noise_factor = noise_factor
        self.probability = probability
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if np.random.random() < self.probability:
            noise = torch.randn_like(tensor) * self.noise_factor
            tensor = tensor + noise
            tensor = torch.clamp(tensor, -2.0, 2.0)
        
        return tensor


class AudioFriendlyTransform:
    """
    Combines multiple audio-specific transformations
    """
    
    def __init__(
        self, 
        use_spec_augment: bool = True, 
        use_noise: bool = True, 
        spec_augment_params: Optional[Dict[str, Any]] = None, 
        noise_params: Optional[Dict[str, Any]] = None
    ):
        self.transforms = []
        
        if use_spec_augment:
            spec_params = spec_augment_params or {}
            self.transforms.append(SpecAugment(**spec_params))
        
        if use_noise:
            noise_params = noise_params or {}
            self.transforms.append(SpectrogramNoise(**noise_params))
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor


class RandomTemporalCrop:
    """
    Crops random temporal segments and resizes back to original size
    Simulates different audio segment lengths
    """
    
    def __init__(self, crop_ratio: float = 0.7):
        self.crop_ratio = crop_ratio
    
    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        crop_width = int(width * self.crop_ratio)
        
        max_start = width - crop_width
        start_x = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
        # Crop the image
        cropped_img = img.crop((start_x, 0, start_x + crop_width, height))
        
        # Resize back to original dimensions
        resized_img = cropped_img.resize((width, height), Image.Resampling.LANCZOS)
        
        return resized_img


# Factory functions for creating augmentation configurations
def create_spec_augment_transform(
    freq_mask_param: int = 10,
    time_mask_param: int = 20,
    num_freq_masks: int = 1,
    num_time_masks: int = 1
) -> AudioFriendlyTransform:
    """Create SpecAugment-only transform"""
    return AudioFriendlyTransform(
        use_spec_augment=True,
        use_noise=False,
        spec_augment_params={
            'freq_mask_param': freq_mask_param,
            'time_mask_param': time_mask_param,
            'num_freq_masks': num_freq_masks,
            'num_time_masks': num_time_masks
        }
    )


def create_noise_transform(
    noise_factor: float = 0.004,
    probability: float = 0.15
) -> AudioFriendlyTransform:
    """Create noise-only transform"""
    return AudioFriendlyTransform(
        use_spec_augment=False,
        use_noise=True,
        noise_params={
            'noise_factor': noise_factor,
            'probability': probability
        }
    )


def create_combined_audio_transform(
    spec_augment_params: Optional[Dict[str, Any]] = None,
    noise_params: Optional[Dict[str, Any]] = None
) -> AudioFriendlyTransform:
    """Create combined audio transform"""
    return AudioFriendlyTransform(
        use_spec_augment=True,
        use_noise=True,
        spec_augment_params=spec_augment_params,
        noise_params=noise_params
    )