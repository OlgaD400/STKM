"""Run STKM on every video in a directory. Return jaccard similarity between true and predicted bboxes."""

import torch
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from stkm.TKM import TKM
import time
import glob, os
from cv_stkm import *
import matplotlib.pyplot as plt
import pandas as pd

############## DEFINE MODEL ##############
#Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Keep the spatial dimensions in the output
model.eval()

# Image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize((224,224)),  # Change this to the size of your images if you want to maintain the same resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

############# Iterate through files ########### 
# min_k = 1
# max_k = 5

annotation_df, video_df = train_json_to_df(json_path='cv_data/vis-train/train.json')
trials = 5
count = 0 

try: 
    cv_df = pd.read_pickle('stgkm_computer_vision_df.pkl')
except:
    cv_df = pd.DataFrame(columns = ['root_filename', 'avg_jaccard_index', 'runtime'])

for dir_name in glob.iglob('cv_data/train/JPEGImages/**/', recursive = True):
    if len(dir_name) == 28:
        #This is the parent directory-skip
        pass
    else:  
        if os.path.isdir(dir_name):

            root_filename = dir_name.split('/')[-2]

            if root_filename in cv_df['root_filename'].values:
                print(dir_name, ' already processed')
                # continue


            relevant_video = video_df[video_df['root_filename'] == root_filename]
    
            if len(relevant_video) == 0:
                print(dir_name, ' No labels available')
                continue

            #There'll be a bug if there's more than one bbox 
            relevant_annotations = annotation_df[annotation_df['video_id'] == relevant_video['id'].values[0]]
            
            #length of tracked objects can vary 
            tracked_objects = relevant_annotations['bboxes'].values

            #for now, if there's more than one tracked ground truth object, ignore the video
            if len(tracked_objects)>1:
                print(dir_name, 'More than one tracked object')
                continue
            
            #Load data
            image_paths, input_data = generate_input_data(dir_name)
            og_image, og_size = get_image(image_path = image_paths[0])
        
            #Run WCSS
            # print('Calculating optimal k')
            # wcss_k = calculate_wcss(input_data = input_data,
            #                         min_k = min_k,
            #                         max_k = max_k,
            #                         max_iter = 100)
            
            
            # opt_num_clusters = np.argmax(np.abs(np.diff(wcss_k))) + 2
            # print(opt_num_clusters)

            print(dir_name, 'Running stkm')

            #Run a handful of trials of stkm
            max_score = 0
            tkm = TKM(input_data)
            for trial in range(trials):
                start_time = time.time()
                tkm.perform_clustering(num_clusters=2, lam=0, max_iter=1000, init_centers = 'kmeans_plus_plus')
                #Evaluate performance
                new_score, true_mask, extended_mask = evaluate_bbox_prediction(true_bboxes = tracked_objects[0],
                                                             weights = tkm.weights,
                                                             og_cols = og_size[0],
                                                             og_rows = og_size[1])
                if new_score>max_score:
                    max_score = new_score
                    runtime = time.time() - start_time
            
            # cv_df = cv_df.append({'root_filename': root_filename,
            #                       'avg_jaccard_index': max_score, 
            #                       'runtime': runtime}, 
            #                       ignore_index = True)

            # cv_df.to_pickle('stgkm_computer_vision_df.pkl')
            
            plt.figure()
            plt.imshow(og_image.resize((224,224)))
            plt.imshow(true_mask, alpha = 0.3)
            plt.imshow(extended_mask, alpha = 0.5)
            plt.show()

            count +=1
            if count%5 == 0:
                break
                print(count)
            
recovered_df = pd.read_pickle('stgkm_computer_vision_df.pkl')
recovered_df.head()

            # plt.figure()
            # plt.imshow(og_image.resize((224,224)))
            # plt.imshow(true_mask, alpha = 0.3)
            # plt.imshow(extended_mask, alpha = 0.5)
            # plt.show()

    

            #let's start with k =2 and then allow k to be preset by the true number of detected objects
            ##?????? 

            #How do I find the optimal number of clusters?
            #How do I confirm that I've found an annotated object -- some degree of overlap
            #How do I confirm that I've found an object that was not annotated?
                #I have to be confident that I'm right as opposed to the OG annotation

            #take the bounding boxes from the train data and compare |num pixels in common|/|num_pixels contained in total|

            #resize bbox to 224 size --> resize 

            # return_masked_image(image_paths = image_paths, index = 0, weights = tkm.weights)

# from cv_stkm import *     
# ann_df, vid_df = train_json_to_df(json_path = 'cv_data/vis-train/train.json')
# image_paths, input_data = generate_input_data(image_directory = 'cv_data/train/JPEGImages/0b5b5e8e5a/')
# return_bbox_image(root_filename='0b5b5e8e5a', image_paths = image_paths, index = 10, annotation_df=ann_df, video_df=vid_df)

