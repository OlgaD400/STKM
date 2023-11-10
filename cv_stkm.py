"Functions for applying stkm to cv problems."
import glob
import json
from typing import Optional, List
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import pandas as pd
from stkm.STKM import STKM

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
layers = list(model.children())[:-2]

model = torch.nn.Sequential(*(layers))  # Keep the spatial dimensions in the output
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


def get_image(image_path: str):
    """
    Open an image from its path.

    Args:
        image_path (str): Path to where image is saved.
    Returns:
        img: Image object.
        og_size (Tuple): Tuple containing original image size (cols, rows).
    """
    img = Image.open(image_path)
    og_size = img.size
    return img, og_size


def image_to_embedding(image_path: Optional[str] = None, img=None):
    """
    Convert an image to a latent space embedding with spatial dimensions
    using a pre-trained ResNet model.

    Args:
        image_path (Optional[str]): Path to imag to be converted to an
            embedding. If not provided, user must instead directly pass
            in an image.
        img: Image object to be converted to an embedding.

    Returns:
        embedding (np.ndarray): Image embedding.
    """

    assert (image_path is not None) | (
        img is not None
    ), "Must pass in either an image or a valid path to the image."

    if img is None:
        img, _ = get_image(image_path)

    img = preprocess(img)
    img = Variable(img.unsqueeze(0))

    with torch.no_grad():
        embedding = model(img)

    return embedding.squeeze(0)  # Remove the batch dimension


def generate_input_data(image_directory: str):
    """
    Generate embeddings for each video frame in a directory.

    Args:
        image_directory (str): Image directory containing frames from a video.
    Returns:
        image_paths (List[str]): All paths to images in the directory.
        input_data (np.ndarray): Array containing embeddings of each video frame in a directory.
    """
    image_paths = [path for path in glob.iglob(image_directory + "*.jpg")]
    image_paths = natsorted(image_paths)
    embeddings = [image_to_embedding(path) for path in image_paths]
    # 2048
    embedding_numpy = [
        embedding.numpy().reshape((2048, -1)) for embedding in embeddings
    ]
    input_data = np.array(embedding_numpy)

    return image_paths, input_data


def return_predicted_masked_image(
    image_paths: List[str], index: int, weights: np.ndarray
):
    """
    Return version of the image masked by predicted clusters from STGkM.

    Args:
        image_paths (List[str]): All paths to video frames in a directory.
        index (int): Index referring to which video frame in the image_path list to visualize.
        weights (np.ndarray): Cluster membership weights from STGkM.

    """
    img = Image.open(image_paths[index])
    img = img.resize((224, 224))
    # 7 7
    mask = np.argmax(weights[index], axis=1).reshape((7, 7))
    # 32 32
    extended_mask = np.repeat(np.repeat(mask, 32, axis=0), 32, axis=1)

    plt.imshow(img)
    plt.imshow(extended_mask, alpha=0.3)
    plt.show()

    return None


def combine_true_bboxes(
    true_bboxes: List[np.ndarray],
    og_cols: int,
    og_rows: int,
    scale_to_grid: bool = False,
):
    """
    Combine all ground truth bboxes into a single foreground mask.

    Args:
        true_bboxes (List[np.ndarray]): Ground truth bboxes. This is a list of shape (num objects,).
            Each array in the list is of length timeslices.
        og_cols (int): Original number of columns in the image.
        og_rows (int): Original number of rows in the image.
        scale_to_grid (bool). Whether or not to scale the ground truth bboxes to a 7x7 grid.
    Returns:
        foreground (np.ndarray): Single foreground mask from bboxes.
    """

    timesteps = len(true_bboxes[0])
    foreground = np.zeros((timesteps, 224, 224))

    # For each object
    for object_bboxes in true_bboxes:
        # For each timestep
        for time in range(timesteps):
            existing_foreground_slice = foreground[time, :, :].copy()
            # object bbox at current timestep
            true_bbox = object_bboxes[time]
            if scale_to_grid:
                true_mask = scaled_mask_from_bbox(
                    true_bbox, og_rows=og_rows, og_cols=og_cols
                )
            else:
                true_mask = true_mask_from_bbox(
                    true_bbox, og_rows=og_rows, og_cols=og_cols
                )
            updated_foreground_slice = np.where(
                true_mask == 1, 1, existing_foreground_slice
            )
            foreground[time, :, :] = updated_foreground_slice.copy()
    return foreground


def true_mask_from_bbox(true_bbox: np.ndarray, og_cols: int, og_rows: int):
    """
    Get 224x224 mask from bbox coordinates.

    Args:
        true_bbox (np.ndarray): Ground truth bbox for a single object.
        og_cols (int): Number of columns in original image.
        og_rows (int): Number of rows in the original image.

    Returns:
        true_mask (np.ndarray): Ground truth bboxes resized to 224x224 image size.
    """
    if true_bbox is None:
        true_mask = np.zeros((224, 224))
    else:
        y_scale = 224 / og_cols
        x_scale = 224 / og_rows

        y = int(np.round(true_bbox[0] * y_scale))
        x = int(np.round(true_bbox[1] * x_scale))
        h = int(np.round(true_bbox[2] * y_scale))
        w = int(np.round(true_bbox[3] * x_scale))
        true_mask = np.zeros((224, 224))
        true_mask[x : x + w, :][:, y : y + h] = 1

    return true_mask


def scaled_mask_from_bbox(true_bbox: np.ndarray, og_cols: int, og_rows: int):
    """
    Scale the true bbox up to a 7x7 grid.

    Args:
        true_bbox (np.ndarray): Ground truth bbox for a single object.
        og_cols (int): Number of columns in original image.
        og_rows (int): Number of rows in the original image.

    Returns:
        scaled_mask (np.ndarray): Ground truth bboxes resized to 7x7 grid size.
    """
    true_mask = true_mask_from_bbox(
        true_bbox=true_bbox, og_cols=og_cols, og_rows=og_rows
    )

    # Scan over the image in blocks of 32x32
    scaled_mask = true_mask.copy()
    for x_scan in range(7):
        for y_scan in range(7):
            if np.any(
                true_mask[
                    x_scan * 32 : (x_scan + 1) * 32, y_scan * 32 : (y_scan + 1) * 32
                ]
                == 1
            ):
                scaled_mask[
                    x_scan * 32 : (x_scan + 1) * 32, y_scan * 32 : (y_scan + 1) * 32
                ] = 1
    return scaled_mask


def evaluate_bbox_prediction(
    true_bboxes: List[np.ndarray],
    weights: np.ndarray,
    og_cols: int,
    og_rows: int,
    scale_to_grid=False,
):
    """
    Calculate the Jaccard index of the true vs predicted bounding boxes.

    Args:
        true_bboxes (List[np.ndarray]): Ground truth bboxes. This is a list of shape (num objects,).
            Each array in the list is of length timeslices.
        weights (np.ndarray): Cluster membership weights from STGkM.
        og_cols (int): Original number of columns in the image.
        og_rows (int): Original number of rows in the image.
        scale_to_grid (bool). Whether or not to scale the ground truth bboxes to a 7x7 grid.
    Returns:
        (float): Average jaccard index of true vs predicted bboxes across over
            all frames of a video.
    """
    timesteps = weights.shape[0]
    # Convert bboxes to single 224x224 foreground mask
    true_masks = combine_true_bboxes(
        true_bboxes, og_cols=og_cols, og_rows=og_rows, scale_to_grid=scale_to_grid
    )

    jaccard_indices = []
    for index in range(timesteps):
        mask = np.argmax(weights[index], axis=1).reshape((7, 7))
        extended_mask = np.repeat(np.repeat(mask, 32, axis=0), 32, axis=1)

        jaccard_index = compare_true_vs_predicted_mask(
            true_mask=true_masks[index], predicted_mask=extended_mask
        )
        jaccard_indices.append(jaccard_index)

    return np.average(jaccard_indices)


def compare_true_vs_predicted_mask(true_mask: np.ndarray, predicted_mask: np.ndarray):
    """
    Compare true vs predicted masks.

    Args:
        true_mask (np.ndarray): Ground truth foreground mask.
        predicted_mask (np.ndarray): Predicted foreground mask.

    Returns:
        max_jaccard_index (float): The jaccard index corresponding to the largest agreement
            between true and predicted foreground mask.
    """
    flat_true_mask = true_mask.reshape(1, -1)
    flat_predicted_bbox = predicted_mask.reshape(1, -1)
    max_jaccard_index = 0
    for cluster in np.unique(flat_predicted_bbox):
        # [1] to take into account the first dimension of flattened masks
        cluster_pixels = np.where(flat_predicted_bbox == cluster)[1]
        # Not where they're equal, but how many values they have in common
        intersection = len(
            np.intersect1d(
                np.nonzero(flat_true_mask)[1], cluster_pixels, assume_unique=True
            )
        )
        # Size of individual masks minus their overlap
        union = (
            np.sum(true_mask == 1) + np.sum(predicted_mask == cluster) - intersection
        )
        jaccard_index = intersection / union

        if jaccard_index > max_jaccard_index:
            max_jaccard_index = jaccard_index

    return max_jaccard_index


def return_bbox_image(
    root_filename: str,
    image_paths: List[str],
    index: int,
    annotation_df: pd.DataFrame,
    video_df: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    scale_to_grid=False,
):
    """
    Return a version of an image masked according to its ground truth and optionally
    also predicted bboxes.

    Args:
        root_filename (str): Root filename of video to be processed.
        image_paths (List[str]): List of all paths corresponding to video frames.
        index (int): Which video frame from image_paths to visualize.
        annotation_df (pd.DataFrame): Dataframe containing ground truth bbox information.
        video_df (pd.DataFrame): Dataframe containing ground truth video information.
        weights (np.ndarray): Cluster membership weights from STGkM.
            If not provided, will not be visualized.
        scale_to_grid (bool). Whether or not to scale the ground truth bboxes to a 7x7 grid.
    """
    img = Image.open(image_paths[index])
    cols, rows = img.size

    bboxes = get_true_bboxes(
        root_filename=root_filename, video_df=video_df, annotation_df=annotation_df
    )
    true_foreground_masks = combine_true_bboxes(
        true_bboxes=bboxes, og_cols=cols, og_rows=rows, scale_to_grid=scale_to_grid
    )
    true_foreground_mask = true_foreground_masks[index]

    if weights is not None:
        pred_mask = np.argmax(weights[index], axis=1).reshape((7, 7))
        extended_pred_mask = np.repeat(np.repeat(pred_mask, 32, axis=0), 32, axis=1)

    plt.imshow(img.resize((224, 224)))
    if weights is not None:
        plt.imshow(extended_pred_mask, alpha=0.5)
    plt.imshow(true_foreground_mask, alpha=0.3)
    plt.show()
    return None


def get_true_bboxes(
    root_filename: str, video_df: pd.DataFrame, annotation_df: pd.DataFrame
):
    """
    Get true bboxes for a given video.

    Args:
        root_filename (str): Root filename of video to be processed.
        annotation_df (pd.DataFrame): Dataframe containing ground truth bbox information.
        video_df (pd.DataFrame): Dataframe containing ground truth video information.

    Returns:
        bboxes (List[np.ndarray]): True bboxes for a given video.
    """
    relevant_video = video_df[video_df["root_filename"] == root_filename]

    if len(relevant_video) == 0:
        print("No labels available.")
        return None

    # There'll be a bug if there's more than one bbox
    relevant_annotations = annotation_df[
        annotation_df["video_id"] == relevant_video["id"].values[0]
    ]
    bboxes = relevant_annotations["bboxes"].values

    return bboxes


def calculate_wcss(input_data: np.ndarray, min_k: int, max_k: int, max_iter=100):
    """Elbow curve."""

    wcss_k = []

    for k in range(min_k, max_k):
        tkm = STKM(input_data)

        tkm.perform_clustering(num_clusters=k, lam=0.8, max_iter=max_iter)

        wcss = np.sum(
            np.linalg.norm(
                tkm.data - tkm.centers @ np.transpose(tkm.weights, axes=[0, 2, 1]),
                2,
                axis=1,
            )
            ** 2
        )

        wcss_k.append(wcss)

    return wcss_k


def train_json_to_df(json_path: str):
    """
    Return multiple dataframes from training json metadata.
    We want bboxes from the anotation df.
    We associate video id between the annotation and video dfs.

    Args:
        json_path (str): Path to json data.
    Returns:
        annotation_df (pd.DataFrame): Dataframe containing bbox information.
        video-df (pd.DataFrame): Dataframe containing video information.
    """

    f = open(json_path)
    data = json.load(f)

    annotation_df = pd.json_normalize(data["annotations"])
    video_df = pd.json_normalize(data["videos"])

    video_df["root_filename"] = video_df.apply(
        lambda row: row["file_names"][0].split("/")[0], axis=1
    )

    return annotation_df, video_df


# dir = 'cv_data/train/JPEGImages/0b34ec1d55/' #0ae1ff65a5/'
# image_paths, input_data = generate_input_data(image_directory=dir)

# tkm = TKM(input_data)
# tkm.perform_clustering(num_clusters = 2, lam = .8, max_iter = 1000)
# return_masked_image(image_paths = image_paths, index = 15, weights = tkm.weights)


# min_k = 1
# max_k = 5
# wcss_k = calculate_wcss(min_k = min_k, max_k = max_k)
# plt.plot(np.arange(min_k, max_k), wcss_k)


# Example usage
# path_prefix = "cv_data/vos-test/JPEGImages/0e1f91c0d7/"
# # path_prefix = "cv_data/plume/fixed_avg/"

# image_paths, input_data = generate_input_data(image_directory = path_prefix)

# print('Data generated.')

# tkm = TKM(input_data)
# start_time = time.time()
# tkm.perform_clustering(num_clusters=2, lam=.8, max_iter=10000, init_centers = 'kmeans_plus_plus')
# runtime = time.time() - start_time


# for frame in range(len(image_paths)):
#     img = Image.open(image_paths[frame])
#     img_t = img.resize((224,224))

#     mask = np.argmax(tkm.weights[frame], axis = 1).reshape((7,7))
#     extended_mask = np.repeat(np.repeat(mask, 32, axis=0), 32, axis=1)

#     plt.imshow(img_t)
#     plt.imshow(extended_mask, alpha = .3)
#     plt.show()
