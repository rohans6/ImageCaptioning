# Download data

# !wget http://images.cocodataset.org/zips/train2014.zip  # Train Data
# !wget http://images.cocodataset.org/zips/val2014.zip    # Val Data
# !wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip # Annotations

# Unzip downloaded files
# unzip train2014.zip
# unzip val2014.zip
# unzip annotations_trainval2014.zip

# Preprocessing Data

# Move all files into data directory
# mkdir data
# mv train2014 data
# mv val2014 data
# mv annotations data

# Import required Libraries
import json
import os
import csv
import pandas as pd
import random
import numpy as np
import gc

# Define variables which will be used in rest of operations.
data_dir="./data"
train_folder="./data/train2014"
val_folder="./data/val2014"
annotations_folder="./data/annotations"

# Process JSON data
train_captions=json.load(open(os.path.join(annotations_folder,"captions_train2014.json"),'r'))
val_captions=json.load(open(os.path.join(annotations_folder,"captions_val2014.json"),'r'))
print(train_captions.keys())
print(train_captions['info'])
print(train_captions['images'][1])
print(train_captions['licenses'][1])
print(train_captions['annotations'][1])
print("No of Images =",len(train_captions['images']))
print("No of Captions =",len(train_captions['annotations']))

trimages = {x['id']: x['file_name'] for x in train_captions['images']}
valimages={x['id']:x['file_name'] for x in val_captions['images']}
data=list()
val=list()
error=list()
for annot in train_captions['annotations']:
  if int(annot['image_id']) in trimages:
    path=os.path.join(train_folder,trimages[int(annot['image_id'])])
    data.append((path,annot['caption']))
  else:
    error.append(("Train",annot['image_id']))

for annot in val_captions['annotations']:
  if int(annot['image_id']) in valimages:
    path=os.path.join(val_folder,valimages[int(annot['image_id'])])
    val.append((path,annot['caption']))
print("No of Training Examples =",len(data))
print("No of Validation Examples =",len(val))
print("No of Errors =",len(error))
random.seed(42)
random.shuffle(data)

# Write csv data
with open(train_folder+"/train.csv",'w') as f:
  writer=csv.writer(f,quoting=csv.QUOTE_ALL)
  writer.writerows(data)

with open(val_folder+"/val.csv","w") as f:
  writer=csv.writer(f,quoting=csv.QUOTE_ALL)
  writer.writerows(val)

