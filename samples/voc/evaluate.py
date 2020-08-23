# Evaluate the mask RCNN model

import os
import sys
import cv2
import numpy as np
from voc_model import VocDataset, InferenceConfig

classes={'aeroplane':1,'bicycle':2,'bird':3,'boat':4,'bottle':5,'bus':6,'car':7,
         'cat':8,'chair':9,'cow':10,'dinningtable':11,'dog':12,'horse':13,'motorbike':14,
         'person':15,'potted_plant':16,'sheep':17,'sofa':18,'train':19,'tv/monitor':20}

objects=['background','aeroplane','bicycle','bird','boat','bottle','bus',
         'car','cat','chair','cow','dinningtable','dog','horse','motorbike',
         'person','potted_plant','sheep','sofa','train','tv/monitor']

ROOT_DIR=os.path.abspath('../..')
sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR=os.path.join(ROOT_DIR,'logs')

from mrcnn import model as modelib, utils

def calc_iou(pred_mask,gt_mask):
    '''
    Calculate the intersection over union between 2 masks.
    Parameters:
    pred_mask: 2D boolean image
    gt_mask: 2D boolean image
    '''
    inter=np.sum(np.logical_and(pred_mask,gt_mask))
    union=np.sum(np.logical_or(pred_mask,gt_mask))
    return inter/union

def compare_image(gt_masks,pred_masks,scores):
    '''
    Compare between ground truth and the prediction of image
    Parameters:
    gt_masks: [H,W,n_objects]: ground truth mask of type bool
    pred_masks: [H,W,n_preds]: prediction mask of type bool
    scores: numpy array, represent the corresponding scores of each prediction
    return: dictionary of tp,fp,fn
    '''
    # Firstly, we consider the simple case
    if(len(gt_masks)==0 and len(pred_masks)==0):
        return {'tp':0,'fp':0,'fn':0}
    elif(len(pred_masks)==0):
        return {'tp':0,'fp':0,'fn':gt_masks.shape[2]}
    elif(len(gt_masks)==0):
        return {'tp':0,'fp':pred_masks.shape[2],'fn':0}

    possible_match=[-1 for i in range(pred_masks.shape[2])]
    for i in range(pred_masks.shape[2]):
        max_iou=0.5
        for j in range(gt_masks.shape[2]):
            iou=calc_iou(pred_masks[:,:,i],gt_masks[:,:,j])
            if(iou>max_iou):
                possible_match[i]=j
    
    # Sort the index of score in descending order
    sort_pred_idx=np.argsort(scores)[::-1]

    # Recalculate the match
    pairs=[]
    mark_gt=[False for i in range(gt_masks.shape[2])] #mark the ground truth, False means ground truth is not selected
    for idx in sort_pred_idx:
        if(possible_match[idx]!=-1):
            if(not mark_gt[possible_match[idx]]):
                pairs.append((idx,possible_match[idx]))
                mark_gt[possible_match[idx]]=True
        
    tp=len(pairs)
    fp=pred_masks.shape[2]-tp
    fn=gt_masks.shape[2]-tp
    return {'tp':tp,'fp':fp,'fn':fn}

def calc_precision_recall(model,data):
    '''
    Calculate the precision and recall for the voc dataset.
    Parameters:
    model: MaskRCNN object defined in mrcnn.model.MaskRCNN
    data: VocDataset object
    '''
    print('Calculating precision and recall ...')
    metrics=[{'tp':0,'fp':0,'fn':0} for i in range(20)]
    for img_id in range(len(data.image_info)):
        gt_masks,gt_class_ids=data.load_mask(img_id)
        img=cv2.imread(data.image_info[img_id]['path'])[:,:,::-1]
        r=model.detect([img])[0]
        pred_masks=r['masks'];scores=r['scores'];pred_class_ids=r['class_ids']
        # All classes in both gt and pred
        class_ids=np.union1d(gt_class_ids,pred_class_ids) 
        for ids in class_ids:
            if(ids in gt_class_ids):
                gt_masks_ids=gt_masks[:,:,gt_class_ids==ids]
            else:
                gt_masks_ids=[]
            
            if(ids in pred_class_ids):
                pred_masks_ids=pred_masks[:,:,pred_class_ids==ids]
                scores_ids=scores[pred_class_ids==ids]
            else:
                pred_masks_ids=[]
                scores_ids=[]
            
            result=compare_image(gt_masks_ids,pred_masks_ids,scores_ids)
            metrics[ids-1]['tp']+=result['tp']
            metrics[ids-1]['fp']+=result['fp']
            metrics[ids-1]['fn']+=result['fn']
        
    presion_recall=[]
    for i in range(20):
        if(metrics[i]['tp']+metrics[i]['fp']==0):precision=0.0
        else:
            precision=metrics[i]['tp']/(metrics[i]['tp']+metrics[i]['fp'])
        if(metrics[i]['tp']+metrics[i]['fn']==0):recall=0.0
        else:
            recall=metrics[i]['tp']/(metrics[i]['tp']+metrics[i]['fn'])
        presion_recall.append({'pre':precision,'rec':recall})
    
    return presion_recall

def create_model(logs,weight_path,confidence,dataset_dir):
    print('Creating model ....')
    config=InferenceConfig()
    config.DETECTION_MIN_CONFIDENCE=confidence
    model=modelib.MaskRCNN(mode='inference',config=config,model_dir=logs)
    model.load_weights(weight_path,by_name=True)
    data=VocDataset()
    data.load_voc(dataset_dir,None)
    return calc_precision_recall(model,data)

def write_to_file(file_name,precision_recall):
    print('Writing to files ...')
    with open(file_name,'w') as f:
        for i in range(20):
            f.write(str(precision_recall[i]['pre'])+' '+str(precision_recall[i]['rec'])+'\n')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Mask RCNN evaluation')
    parser.add_argument('--logs',required=False,default=DEFAULT_LOGS_DIR,metavar='path/to/logs',help='logs')
    parser.add_argument('--weight_path',required=True,metavar='path/to/weight_file',help='weight file')
    parser.add_argument('--confidence', required=True,metavar='confidence score',help='confidence to be an object')
    parser.add_argument('--data_dir',required=True,metavar='path/to/validation/set',help='path to validation')

    args=parser.parse_args()
    precision_recall=create_model(args.logs,args.weight_path,float(args.confidence),args.data_dir)

    file_name='eval/eval_{}.txt'.format(str(int(100*float(args.confidence))))
    write_to_file(file_name,precision_recall)
