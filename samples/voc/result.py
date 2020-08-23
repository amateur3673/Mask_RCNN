import numpy as np
import matplotlib.pyplot as plt
from voc_model import objects
class_ids=[6,7,11,12,14]
n_indexes=(95-10)//5+1
precision_all=np.zeros((20,n_indexes))
recall_all=np.zeros((20,n_indexes))
idx=0
for score in range(95,5,-5):
    with open('eval/eval_{}.txt'.format(score),'r') as f:
        for i in range(20):
            data=f.readline().split(' ')
            precision_all[i][idx]=float(data[0])
            recall_all[i][idx]=float(data[1])
    idx+=1

precisions=precision_all[class_ids]
recalls=recall_all[class_ids]

result={}
for i in range(len(class_ids)):
    result[i]={'precision':list(precisions[i]),'recall':list(recalls[i])}

for i in range(len(class_ids)):
    result[i]['precision'].insert(0,1.0)
    result[i]['recall'].insert(0,0.0)
    for j in range(len(result[i]['precision'])):
        result[i]['precision'][j]=max(result[i]['precision'][j:])

color=['b','g','r','c','m','y','k','w']
for i in range(len(class_ids)):
    plt.plot(result[i]['recall'],result[i]['precision'],color=color[i],label=objects[class_ids[i]+1])
plt.legend(loc='upper right')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Refine AP Curve')
plt.show()

def calc_ap(result_id):
    '''
    Calculate the AP of a class
    '''
    ap=0
    for i in range(1,len(result_id['precision'])):
        ap+=0.5*(result_id['precision'][i]+result_id['precision'][i-1])*(result_id['recall'][i]-result_id['recall'][i-1])
    return ap

ap=[]
for i in range(len(class_ids)):
    ap.append(calc_ap(result[i]))

print('MAP is: {}'.format(np.mean(np.array(ap))))