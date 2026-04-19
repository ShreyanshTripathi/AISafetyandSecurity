import torch
from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import numpy as np
import itertools
import os
from sklearn.metrics import jaccard_score

# --- Folders and Paths ---
gradcam_mask_dir = (
    "/content/masks_gradcam"  # Directory with saved Grad-CAM masks (*.png)
)
image_paths = [
    # '/content/n01443537_goldfish.JPEG',
    #  '/content/n01491361_tiger_shark.JPEG',
    "/content/n01608432_kite.JPEG",
    #  '/content/n01616318_vulture.JPEG',
    #  '/content/n01677366_common_iguana.JPEG',
    #  '/content/n02007558_flamingo.JPEG',
    #  '/content/n02018207_American_coot.JPEG',
    #  '/content/n02098286_West_Highland_white_terrier.JPEG',
    #  '/content/n04037443_racer.JPEG',
    #  '/content/n07747607_orange.JPEG'
]  # Your image list

# --- Model and Preprocessing (same as before) ---
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
model.eval()
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def batch_predict(images):
    imgs = []
    for img in images:
        pil_img = Image.fromarray(img.astype(np.uint8))
        tensor = preprocess(pil_img).to(device)
        imgs.append(tensor)
    batch = torch.stack(imgs)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


# --- Tuning Grid ---
param_grid = {"num_samples": [500], "num_features": [60]}
all_param_combos = list(itertools.product(*param_grid.values()))

# --- Tuning Loop ---
results = []
for img_path in image_paths:
    basename = os.path.splitext(os.path.basename(img_path))[0]
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)

    # Load and resize Grad-CAM mask to image size, binarize
    gradcam_mask_path = os.path.join(f"{basename}_Grad-CAM_mask.png")
    gradcam_mask = np.array(
        Image.open(gradcam_mask_path).resize((image_np.shape[1], image_np.shape[0]))
    )
    gradcam_mask_binary = (gradcam_mask > 128).astype(int).flatten()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np, batch_predict, top_labels=1, hide_color=0, num_samples=1000
    )
    top_label = explanation.top_labels[0]

    for num_samples, num_features in all_param_combos:
        explanation = explainer.explain_instance(
            image_np, batch_predict, top_labels=1, hide_color=0, num_samples=num_samples
        )
        img_with_mask, lime_mask = explanation.get_image_and_mask(
            top_label, positive_only=True, num_features=num_features, hide_rest=False
        )
        # Binarize LIME mask
        lime_mask_bin = (lime_mask > 0).astype(int).flatten()
        # Compute IoU (Jaccard) with Grad-CAM mask
        iou = jaccard_score(gradcam_mask_binary, lime_mask_bin)
        results.append(
            {
                "image": basename,
                "num_samples": num_samples,
                "num_features": num_features,
                "IoU_with_GradCAM": iou,
            }
        )

import pandas as pd

results_df = pd.DataFrame(results)
print(results_df.sort_values(by="IoU_with_GradCAM", ascending=False).head(10))
