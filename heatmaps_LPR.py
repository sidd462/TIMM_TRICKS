from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from baselines.ViT.ViT_LRP import vit_large_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
CLS2IDX={0:"Bacterial_Pneumonia_segmented",
         1:"Covid_segmented", 
         2:"Normal_segmented",
         3:"Tubercolusis",
         4:"Viral_Pneumonia_segmented"}
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
    normalize,
])
# Initialize ViT pretrained model
model = vit_LRP(num_classes=5).cuda()
state_dict = torch.load('vitLargeFold_1.pth')
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# LRP attribution generator
attribution_generator = LRP(model)
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    # vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR) #! uncommnet this when you want to display using mathplot as mathplot want rgb but when saving with cv2 it want bgr
    return vis

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Assuming the following functions and model initialization from your previous code:
# - show_cam_on_image
# - generate_visualization
# - model and normalization initialization

def process_folder(segmented_images_folder, mask_folder, output_folder):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List all files in the segmented_images_folder
    segmented_images = sorted(os.listdir(segmented_images_folder))
    mask_images = sorted(os.listdir(mask_folder))
    
    for img_name, mask_name in zip(segmented_images, mask_images):
        # Construct the full path for each image and mask
        img_path = os.path.join(segmented_images_folder, img_name)
        mask_path = os.path.join(mask_folder, mask_name)
        
        # Load and transform the image
        image = Image.open(img_path)
        transformed_image = transform(image)
        
        # Generate heatmap visualization
        heatmap = generate_visualization(transformed_image)
        
        # Load and resize the mask to match heatmap
        mask_image = Image.open(mask_path).convert('L')  # Ensure mask is in grayscale
        mask_image = mask_image.resize((224, 224))  # Resize mask to match heatmap
        mask = np.array(mask_image)
        mask = mask.astype(np.float32) / 255  # Normalize mask
        mask = np.stack((mask,)*3, axis=-1)  # Ensure mask is the correct shape (224,224,3)
        
        # Apply the mask to the heatmap visualization
        masked_heatmap = heatmap * mask  # This will broadcast correctly
        
        # Save the masked heatmap
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, masked_heatmap)

# Example usage:
segmented_images_folder = 'Bacterial_Pneumonia_segmentation'
mask_folder = 'Bacterial_Pneumonia_mask'
output_folder = 'temp2'

# Call the function
process_folder(segmented_images_folder, mask_folder, output_folder)
