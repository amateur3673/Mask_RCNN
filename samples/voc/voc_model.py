'''
Mask RCNN for VOC datasets
'''

import os
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

classes={'aeroplane':1,'bicycle':2,'bird':3,'boat':4,'bottle':5,'bus':6,'car':7,
         'cat':8,'chair':9,'cow':10,'dinningtable':11,'dog':12,'horse':13,'motorbike':14,
         'person':15,'potted_plant':16,'sheep':17,'sofa':18,'train':19,'tv/monitor':20}

objects=['background','aeroplane','bicycle','bird','boat','bottle','bus',
         'car','cat','chair','cow','dinningtable','dog','horse','motorbike',
         'person','potted_plant','sheep','sofa','train','tv/monitor']
ROOT_DIR=os.path.abspath("../../") # Root directory
# Change the sys path to import Mask RCNN as usual
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modelib, utils
from mrcnn.visualize import apply_mask,random_colors
#Pretrained COCO weights

COCO_MODEL_PATH=os.path.join(ROOT_DIR,'mask_rcnn_coco.h5')

# Default directory to save logs

DEFAULT_LOGS_DIR=os.path.join(ROOT_DIR,'logs')

def read_file(xml_files):
    in_file=open(xml_files)
    tree=ET.parse(in_file)
    root=tree.getroot()
    annotate=[]
    for obj in root.iter('object'):
        difficult=obj.find('difficult').text
        class_label=obj.find('name').text
        if(class_label not in classes.keys() or int(difficult)==1):continue
        class_id=classes[class_label]
        boxes=obj.find('bndbox')
        bbox=(int(boxes.find('xmin').text),int(boxes.find('ymin').text),int(boxes.find('xmax').text),int(boxes.find('ymax').text))
        dict_box={'class_id':class_id,'bbox':bbox}
        annotate.append(dict_box)
    return annotate

def extract_mask(segment_img,bbox):
    '''
    Extract a mask from segmentation image with bounding box
    Params:
    segment_img: image of height,width,3
    bbox: list of left,top,right,bottom
    '''
    mask=np.zeros((segment_img.shape[0],segment_img.shape[1])).astype(bool)
    left,top,right,bottom=bbox
    # Extract a window to cover the object (label for detection)
    window=segment_img[top:bottom,left:right]
    flat_window=np.reshape(window,(window.shape[0]*window.shape[1],3))
    # Extract color in that window
    color=np.unique(flat_window,axis=0)
    # Remove the background color in the color list
    if(np.array([0,0,0]) in color):
        color=np.delete(color,np.array([0,0,0]),axis=0)
    # Since in the window, there maybe more than 2 non-background colors, we need to carefully choose the
    # main color represents the object in the box
    count=np.zeros((color.shape[0],))
    pos=np.zeros((window.shape[0],window.shape[1],color.shape[0]))
    for i in range(color.shape[0]):
        same_pos=window==color[i]
        pos[:,:,i]=same_pos[:,:,0]*same_pos[:,:,1]*same_pos[:,:,2]
        count[i]=len(np.where(pos[:,:,i]==True)[0])
    
    best_color=np.argmax(count)
    mask[top:bottom,left:right]=pos[:,:,best_color]
    return mask

class VOC_Config(Config):
    '''
    Config for training Pascal VOC dataset
    Overrides the Config class
    '''
    NAME='voc'

    # Define the batch size, just use 2 images for a single GPU
    IMAGES_PER_GPU=2

    #Number of classes in the VOC datasets, here we just use a small of classes
    NUM_CLASSES=1+len(classes.keys())
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


class InferenceConfig(VOC_Config):
    '''
    Config class for inference
    '''
    GPU_COUNT=1
    IMAGES_PER_GPU=1

class VocDataset(utils.Dataset):
    '''
    Subclass the utils.Dataset class. Define the Pascal VOC dataset
    '''
    def load_voc(self,dataset_dir,subset):
        '''
        Load the VOC dataset
        '''
        for name in classes.keys():
            self.add_class('voc',classes[name],name)
        
        # Segmentation directory
        self.segmentation_path=os.path.join(dataset_dir,"SegmentationObject")
        # Annotation directory
        annotation_path=os.path.join(dataset_dir,"Annotations")
        # list of images
        list_images=os.listdir(self.segmentation_path)
        image_path=os.path.join(dataset_dir,"JPEGImages")
        for image in list_images:
            image_id=image[:-3] #Take the image name for image id
            img=cv2.imread(self.segmentation_path+'/'+image)
            height,width,_=img.shape
            annotation=read_file(annotation_path+'/'+image_id+'xml')
            self.add_image(
                "voc",
                image_id=image_id,path=image_path+'/'+image_id+'jpg',
                width=width,height=height,box_info=annotation)

    def load_mask(self,image_id):
        '''
        Overrides the load_mask method
        '''
        image_info=self.image_info[image_id]
        annotation=image_info['box_info'] #annotation of the image
        if(image_info['source']!='voc'):
            return super(self.__class__,self).load_mask(image_id)
        
        # Read the Segmentation Image
        img=cv2.imread(self.segmentation_path+'/'+image_info['id']+'png')[:,:,::-1]
        # Initialize the mask
        mask=np.zeros((image_info['height'],image_info['width'],len(annotation))).astype(bool)
        # Initialize the class corresponding to each mask
        class_id=np.zeros((len(annotation))).astype(np.int32)

        for i in range(mask.shape[-1]):
            class_obj=annotation[i]['class_id']
            mask[:,:,i]=extract_mask(img,annotation[i]['bbox'])
            class_id[i]=class_obj
        return mask,class_id
    def image_reference(self,image_id):
        info=self.image_info[image_id]
        if(info['source']=='voc'):
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)         
def train(model):
    '''
    Train the model of VOC dataset
    '''
    # Train dataset
    dataset_train=VocDataset()
    dataset_train.load_voc(args.train_data,None)
    dataset_train.prepare()
    # Validation set
    dataset_val=VocDataset()
    dataset_val.load_voc(args.val_data,None)
    dataset_val.prepare()

    print('Training network:')
    model.train(dataset_train,dataset_val,learning_rate=config.LEARNING_RATE,epochs=10,layers='5+')

def draw_box(image_path,model):
    '''
    Get the result and draw the bounding box, as well as the mask for each object in image
    '''
    img=cv2.imread(image_path)[:,:,::-1]
    r=model.detect([img])[0]
    rois=r['rois']
    class_ids=r['class_ids']
    scores=r['scores']
    masks=r['masks']
    N=masks.shape[2]
    # First we will color each mask of object
    mask_image=img.astype(np.uint32).copy()
    colors=random_colors(N)
    for i in range(N):
        mask=masks[:,:,i]
        color=colors[i]
        mask_image=apply_mask(mask_image,mask,color)
    
    mask_image=mask_image.astype(np.uint8)
    # We need to draw the rectangle in the image
    image=Image.fromarray(mask_image,mode='RGB')
    draw=ImageDraw.Draw(image)
    for i in range(N):
        y1,x1,y2,x2=rois[i] #get the coordinate
        score=scores[i]
        class_id=class_ids[i]
        text=objects[class_id]+':'+str(round(score,2))
        draw.rectangle(((x1,y1),(x2,y2)))
        draw.text((x1,y1),text)
    return image


if __name__=='__main__':
   import argparse

   #Parse the command line
   parser=argparse.ArgumentParser(description='Mask RCNN for VOC dataset')
   parser.add_argument("command",metavar="<command>",help="'train' or 'inference'")
   parser.add_argument("--train_data",required=False,metavar='path/to/train/dataset',help='Directory of train data')
   parser.add_argument("--val_data",required=False,metavar='path/to/val/dataset',help='Directory of validation data')
   parser.add_argument("--weights",required=False,metavar='path/to/weights/file',help='Weights file for inference')
   parser.add_argument("--logs",required=False,metavar='path/to/log',help='logs',default=DEFAULT_LOGS_DIR)
   parser.add_argument("--image",required=False,metavar='path/to/image',help='inference image')
   parser.add_argument("--name",required=False,metavar='image/name',help='detection image name')

   args=parser.parse_args()
   print(args.logs)

   if(args.command=='train'):
       assert args.train_data, "Argument --train_data is required for training"
       assert args.val_data, "Argument --val_data is required for training"
   else:
       assert args.weights, "Weights must be provided for inference"
       assert args.image, "Image must be provided in inference mode"
    
   if(args.command=='train'):
       config=VOC_Config()
   else:
       config=InferenceConfig()
   config.display()
   #Build the model
   if(args.command=='train'):
       print('Create training model')
       model=modelib.MaskRCNN(mode='training',config=config,model_dir=args.logs)

   elif(args.command=='inference'):
       model=modelib.MaskRCNN(mode='inference',config=config,model_dir=args.logs)

   if(args.weights.lower()=='coco'):
       weight_path=COCO_MODEL_PATH
   else:
       weight_path=args.weights

   print('Loading weights from coco model path:')
   
   if(args.weights.lower()=='coco'):
        print('Load coco model')
        model.load_weights(weight_path,by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                                             "mrcnn_bbox", "mrcnn_mask"])
   
   else:
       model.load_weights(weight_path,by_name=True)
   if(args.command=='train'):
       train(model)
       model.keras_model.save_weights('/content/voc.h5')
   elif(args.command=='inference'):
       image=draw_box(args.image,model)
       image.save(args.name)
