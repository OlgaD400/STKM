"""Example script for running stkm for cv problems."""

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import models, transforms
from stkm.STKM import STKM
from cv_stkm import generate_input_data, return_predicted_masked_image

############## DEFINE MODEL ##############
# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)

model = torch.nn.Sequential(
    *(list(model.children())[:-2])
)  # Keep the spatial dimensions in the output
model.eval()

# Image transformation pipeline
preprocess = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),  # Change this to the size of your images if you want to maintain the same resolution
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        # transforms.ColorJitter(),
    ]
)
##############  GENERATE EMBEDDINGS ##############

dir = "cv_data/plume/LeakVid78Frames/"
image_paths, input_data = generate_input_data(
    image_directory=dir, model=model, preprocess=preprocess
)
print("Data generated. Running STKM.")

##############  RUN STkM ##############

NUM_CLUSTERS = 4
stkm = STKM(input_data[::10, :, :])
stkm.perform_clustering(num_clusters=NUM_CLUSTERS, lam=0.8, max_iter=1000, method="L2")

fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
ALPHA = 0.50
for time in range(8):
    img, extended_mask, full_geoms = return_predicted_masked_image(
        image_paths=image_paths,
        index=time * 5,
        weights=stkm.weights,
        plot_outlines=True,
        plot=False,
    )
    for full_geom in full_geoms:
        x, y = full_geom.exterior.xy
        axs[time // 4][time % 4].plot(x, y, linewidth=2, color="r")
    axs[time // 4][time % 4].imshow(img)
    axs[time // 4][time % 4].imshow(extended_mask, alpha=ALPHA)
    axs[time // 4][time % 4].title.set_text("Time %d" % (time * 5))
plt.savefig("predicted_plume_3.pdf", format="pdf")
