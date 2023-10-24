import torch

from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from stkm.TKM import TKM
import time
import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import json
import pandas as pd
from typing import Optional



# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
print(list(model.children())[-3][0])
layers = list(model.children())[:-2] +torch.nn.Sequential(*list(model.children())[-3][0])

print(layers)
model = torch.nn.Sequential(*(layers))  # Keep the spatial dimensions in the output
model.eval()

# Image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize((224,224)),  # Change this to the size of your images if you want to maintain the same resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image(image_path):
    img = Image.open(image_path)
    og_size = img.size
    return img, og_size

def image_to_embedding(image_path: Optional[str] = None, img = None):
    """Convert an image to a latent space embedding with spatial dimensions using a pre-trained ResNet model."""
    
    assert (image_path is not None) | (img is not None), "Must pass in either an image or a valid path to the image."
    
    if img is None: 
        img, _ = get_image(image_path)
    
    img = preprocess(img)
    img = Variable(img.unsqueeze(0))

    with torch.no_grad():
        embedding = model(img)

    return embedding.squeeze(0)# Remove the batch dimension

def generate_input_data(image_directory):
    """
    Generate embeddings for each video frame in a directory.     
    """
    image_paths = [path for path in glob.iglob(image_directory + '*.jpg')]
    image_paths= natsorted(image_paths)
    embeddings = [image_to_embedding(path) for path in image_paths]
    print(embeddings[0].shape)
    #2048
    embedding_numpy = [embedding.numpy().reshape((2048, -1)) for embedding in embeddings]
    input_data = np.array(embedding_numpy)

    return image_paths, input_data

def return_masked_image(image_paths, index, weights):
    """
    Return a masked version of the image, color coded by cluster.
    """
    img = Image.open(image_paths[index])
    img = img.resize((224,224))
    #7 7 
    mask = np.argmax(weights[index], axis = 1).reshape((7,7))
    #32 32
    extended_mask = np.repeat(np.repeat(mask, 32, axis=0), 32, axis=1)

    plt.imshow(img)
    plt.imshow(extended_mask, alpha = .3)
    plt.show()

    return None

def evaluate_bbox_prediction(true_bboxes, weights, og_cols, og_rows):
    """
    Evaluate the performance of stkm
    """
    jaccard_indices = []
    for index, true_bbox in enumerate(true_bboxes):
        mask = np.argmax(weights[index], axis = 1).reshape((7,7))
        extended_mask = np.repeat(np.repeat(mask, 32, axis=0), 32, axis=1)

        jaccard_index, true_mask = compare_true_vs_predicted_bboxes(true_bbox = true_bbox,
                                         predicted_bbox = extended_mask,
                                         og_cols = og_cols,
                                         og_rows = og_rows)
        jaccard_indices.append(jaccard_index)
    
    return np.average(jaccard_indices), true_mask, extended_mask

def compare_true_vs_predicted_bboxes(true_bbox, predicted_bbox, og_cols, og_rows):
    """
    Resize the true object bounding boxes and compare them to the predicted bounding boxes
    """
    # img = img.resize((224,224), resample = Image.BILINEAR)
    #
    #If no object is tracked for that time step
    if true_bbox is None:
        true_mask = np.zeros((224,224))
    else:
        y_scale = 224/og_cols
        x_scale = 224/og_rows

        y = int(np.round(true_bbox[0]*y_scale))
        x = int(np.round(true_bbox[1]*x_scale))
        h = int(np.round(true_bbox[2]*y_scale))
        w = int(np.round(true_bbox[3]*x_scale))
        true_mask = np.zeros((224,224))
        true_mask[x: x+w,:][:, y:y+h] = 1
    
    flat_true_mask = true_mask.reshape(1,-1)
    flat_predicted_bbox = predicted_bbox.reshape(1,-1)
    max_jaccard_index = 0
    for cluster in np.unique(flat_predicted_bbox):
        #[1] to take into account the first dimension of flattened masks
        cluster_pixels = np.where(flat_predicted_bbox == cluster)[1]
        #Not where they're equal, but how many values they have in common
        intersection = len(np.intersect1d(np.nonzero(flat_true_mask)[1], cluster_pixels, assume_unique = True))
        #Size of individual masks minus their overlap
        union = np.sum(true_mask == 1) + np.sum(predicted_bbox == cluster) - intersection
        jaccard_index = intersection/union

        if jaccard_index > max_jaccard_index:
            max_jaccard_index = jaccard_index

    return max_jaccard_index, true_mask


def return_bbox_image(root_filename, image_paths, index, annotation_df, video_df, resize = False):
    """
    Return a version of an image masked according to its ground truth bboxes.
    """
    relevant_video = video_df[video_df['root_filename'] == root_filename]
    
    if len(relevant_video) == 0:
        print('no labels available')
        return None
    
    #There'll be a bug if there's more than one bbox 
    relevant_annotations = annotation_df[annotation_df['video_id'] == relevant_video['id'].values[0]]
    bbox = relevant_annotations['bboxes'].values[0][index]
    
    if bbox[0] is None:
        print('No object detected')
        return None
    
    img = Image.open(image_paths[index])
    cols, rows = img.size
    
    if resize:
        img = img.resize((224,224), resample = Image.BILINEAR)

        y_scale = 224/cols
        x_scale = 224/rows

        y = int(np.round(bbox[0]*y_scale))
        x = int(np.round(bbox[1]*x_scale))
        h = int(np.round(bbox[2]*y_scale))
        w = int(np.round(bbox[3]*x_scale))
        mask = np.zeros((224,224))
    else:
        y = int(bbox[0])
        x = int(bbox[1])
        h = int(bbox[2])
        w = int(bbox[3])
        mask = np.zeros((720, 1280))
    
    mask[x: x+w,:][:, y:y+h] = 1

    plt.imshow(img)
    plt.imshow(mask, alpha = 0.3)
    plt.show()

    return None

def return_masked_annotation(annotation_path, weights):
    """
    Return a masked version of the annotation, color coded by cluster.
    """

    img = Image.open(annotation_path)
    img = img.resize((224, 224))
    mask = np.argmax(weights[0])
    extended_mask = np.repeat(np.repeat(mask, 32, axis = 0), 32, axis = 1)

    plt.imshow(img)
    plt.imshow(extended_mask, alpha = .3)
    plt.show()

    return None


def calculate_wcss(input_data, min_k, max_k, max_iter= 100):
    """Elbow curve."""

    wcss_k = []

    for k in range(min_k, max_k):
        tkm = TKM(input_data)

        tkm.perform_clustering(num_clusters=k, lam=.8, max_iter=max_iter)
        
        wcss = np.sum(np.linalg.norm(
            tkm.data - tkm.centers @ np.transpose(
                tkm.weights, axes=[0, 2, 1]),
            2,
            axis=1,
            )** 2)
        
        wcss_k.append(wcss)
    
    return wcss_k

def train_json_to_df(json_path:str):
    """
    Return multiple dataframes from training json metadata. 

    We want bboxes from the anotation df. 
    We associate video id between the annotation and video dfs. 
    """

    f = open(json_path)
    data = json.load(f)

    annotation_df = pd.json_normalize(data['annotations'])
    video_df = pd.json_normalize(data['videos'])

    video_df['root_filename'] = video_df.apply(lambda row: row['file_names'][0].split('/')[0], axis = 1)

    return annotation_df, video_df

dir = 'cv_data/train/JPEGImages/0b34ec1d55/' #0ae1ff65a5/'
image_paths, input_data = generate_input_data(image_directory=dir)

tkm = TKM(input_data)
tkm.perform_clustering(num_clusters = 2, lam = .8, max_iter = 1000)
return_masked_image(image_paths = image_paths, index = 15, weights = tkm.weights)



#min_k = 1
#max_k = 5
#wcss_k = calculate_wcss(min_k = min_k, max_k = max_k)
#plt.plot(np.arange(min_k, max_k), wcss_k)


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
