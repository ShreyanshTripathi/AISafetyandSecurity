import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from pytorch_grad_cam import GradCAM, AblationCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_paths = ['/content/n01443537_goldfish.JPEG',
               '/content/n01491361_tiger_shark.JPEG',
               '/content/n01608432_kite.JPEG',
               '/content/n01616318_vulture.JPEG',
               '/content/n01677366_common_iguana.JPEG',
               '/content/n02007558_flamingo.JPEG',
               '/content/n02018207_American_coot.JPEG',
               '/content/n02098286_West_Highland_white_terrier.JPEG',
               '/content/n04037443_racer.JPEG',
               '/content/n07747607_orange.JPEG']  # Replace with your images

images = []
input_tensors = []
for img_path in image_paths:
    img = Image.open(img_path).convert('RGB')
    img_arr = np.array(img).astype(np.float32) / 255.0
    images.append(img_arr)
    input_tensors.append(transform(img))
input_batch = torch.stack(input_tensors).to(device)

# Directories for overlays and masks
save_dirs_overlay = {
    'Grad-CAM': 'results_gradcam',
    'AblationCAM': 'results_ablationcam',
    'ScoreCAM': 'results_scorecam'
}
save_dirs_mask = {
    'Grad-CAM': 'masks_gradcam',
    'AblationCAM': 'masks_ablationcam',
    'ScoreCAM': 'masks_scorecam'
}
for v in list(save_dirs_overlay.values()) + list(save_dirs_mask.values()):
    os.makedirs(v, exist_ok=True)

target_layers = [model.layer4[-1]]
cam_methods = {
    'Grad-CAM': GradCAM(model=model, target_layers=target_layers),
    'AblationCAM': AblationCAM(model=model, target_layers=target_layers),
    'ScoreCAM': ScoreCAM(model=model, target_layers=target_layers)
}

results = {method: [] for method in cam_methods}

for i in range(len(images)):
    input_tensor = input_batch[i].unsqueeze(0).float()
    output = model(input_tensor)
    target_category = int(output.argmax(dim=1).item())

    for name, cam in cam_methods.items():
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        h, w = images[i].shape[:2]
        resized_cam = cv2.resize(grayscale_cam, (w, h))
        # Save mask (scale to 0-255 for uint8)
        mask_img = (resized_cam * 255).astype(np.uint8)
        mask_save_path = os.path.join(
            save_dirs_mask[name],
            f"{os.path.splitext(os.path.basename(image_paths[i]))[0]}_{name}_mask.png"
        )
        Image.fromarray(mask_img).save(mask_save_path)
        # Save overlay
        visualization = show_cam_on_image(images[i], resized_cam, use_rgb=True)
        results[name].append(visualization)
        overlay_save_path = os.path.join(
            save_dirs_overlay[name],
            f"{os.path.splitext(os.path.basename(image_paths[i]))[0]}_{name}.png"
        )
        Image.fromarray(visualization).save(overlay_save_path)

# (Optional) Side-by-side visualization in notebook
for idx, path in enumerate(image_paths):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Explanations for Image: {path}", fontsize=18)
    for j, (name, vis_list) in enumerate(results.items()):
        axs[j].imshow(vis_list[idx])
        axs[j].set_title(name, fontsize=16)
        axs[j].axis('off')
    plt.tight_layout()
    plt.show()
