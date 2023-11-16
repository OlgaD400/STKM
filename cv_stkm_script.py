"""Run STKM on every video in a directory. Return jaccard similarity between true and predicted bboxes."""
import time
import glob, os
import torch
from torchvision import models, transforms
import pandas as pd
from stkm.STKM import STKM
from cv_stkm import (
    get_true_bboxes,
    train_json_to_df,
    generate_input_data,
    get_image,
    evaluate_bbox_prediction,
)

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
    ]
)

############# Iterate through files ###########
annotation_df, video_df = train_json_to_df(json_path="cv_data/vis-train/train.json")
TRIALS = 5
COUNT = 0
RESULTS_FILE = "stgkm_computer_vision_df_L1.pkl"

try:
    print("Attempting to load existing experimental results.")
    cv_df = pd.read_pickle(RESULTS_FILE)
except:
    print("Creating new dataframe to store experiment results.")
    cv_df = pd.DataFrame(columns=["root_filename", "avg_jaccard_index", "runtime"])

for dir_name in glob.iglob("cv_data/train/JPEGImages/**/", recursive=True):
    if len(dir_name) == 28:
        # This is the parent directory-skip
        pass
    else:
        if os.path.isdir(dir_name):
            root_filename = dir_name.split("/")[-2]

            if root_filename in cv_df["root_filename"].values:
                print(dir_name, " already processed")
                continue

            tracked_objects = get_true_bboxes(
                root_filename=root_filename,
                annotation_df=annotation_df,
                video_df=video_df,
            )

            if tracked_objects is None:
                print(dir_name, " No labels available")
                continue

            # Load data
            image_paths, input_data = generate_input_data(dir_name)
            og_image, og_size = get_image(image_path=image_paths[0])

            print(dir_name, "Running stkm")

            # Run a handful of trials of stkm
            MAX_SCORE = 0
            tkm = STKM(input_data)
            for trial in range(TRIALS):
                start_time = time.time()
                tkm.perform_clustering(
                    num_clusters=2,
                    lam=0.8,
                    max_iter=200,
                    init_centers="kmeans_plus_plus",
                    method="L1",
                    gamma=1e-4,
                )
                # Evaluate performance
                new_score = evaluate_bbox_prediction(
                    true_bboxes=tracked_objects,
                    weights=tkm.weights,
                    og_cols=og_size[0],
                    og_rows=og_size[1],
                    scale_to_grid=True,
                )
                if new_score > MAX_SCORE:
                    MAX_SCORE = new_score
                    runtime = time.time() - start_time

            cv_df = cv_df.append(
                {
                    "root_filename": root_filename,
                    "avg_jaccard_index": MAX_SCORE,
                    "runtime": runtime,
                },
                ignore_index=True,
            )

            cv_df.to_pickle(RESULTS_FILE)
            print("Max score", MAX_SCORE)
            COUNT += 1
            if COUNT % 100 == 0:
                print(COUNT)

recovered_df = pd.read_pickle(RESULTS_FILE)
recovered_df.head()
