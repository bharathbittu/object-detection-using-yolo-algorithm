import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import cv2
from PIL import Image,ImageDraw,ImageFont
from keras.models import load_model,Model
import tensorflow as tf
import random
import colorsys
from keras import backend as K

model=load_model('yolo.h5')
image = Image.open('test2.jpg')
#imd=cv2.imread('test1.jpg')
resized_image = image.resize(tuple(reversed((608,608))), Image.BICUBIC)
image_data = np.array(resized_image, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0) 
k=model.predict(image_data)
feats=k
with open('coco_classes.txt') as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]

with open('yolo_anchors.txt') as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
anchors_tensor =anchors.reshape(1, 1, num_anchors, 2)
#reshaping 425 features to 5 anchors
feats=feats.reshape(19,19,5,85)
box_confidence = K.sigmoid(feats[..., 4:5])
box_xy = K.sigmoid(feats[..., 0:2])
box_wh = K.exp(feats[..., 2:4])
box_class_probs = K.softmax(feats[..., 5:])
#cinvert tensor to numpy array
'''with tf.Session() as sess:
    box_confidence=sess.run(box_confidence)
    box_xy=sess.run(box_xy)
    box_wh=sess.run(box_wh)
    box_class_probs=sess.run(box_class_probs)'''

#we will get the points w.r.t the 19x19grids so to convert them to 1x1 grid we need add c array and divide it by 19.0
c=np.zeros(361*2)
t=0
for i in range(19):
    for j in range(19):
        c[t]=j
        c[t+1]=i
        t=t+2
#reshaping c so that we can add it to box_xy to make it1x1
c=c.reshape(19,19,1,2)
conv_dims=19.0
    
    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
box_xy = (box_xy + c) / conv_dims
box_wh = box_wh * anchors_tensor / conv_dims
#so we got middle points of the boxes and widths,heights
#by cnsideringboth box confidences and class confidenceswe need to detect

box_scores=np.multiply(box_confidence,box_class_probs)
classes=K.argmax(box_scores)
classes_score=K.max(box_scores,axis=-1)
#masking the values below threshold

mask=K.greater_equal(classes_score,0.1)

scores=tf.boolean_mask(classes_score,mask)
box_xy=tf.boolean_mask(box_xy,mask)
box_wh=tf.boolean_mask(box_wh,mask)
classes=tf.boolean_mask(classes,mask)
box_confidence=tf.boolean_mask(box_confidence,mask)

c=image.size
#now convert boxxy and box wh to box_corners
box_left_cor=(box_xy-(box_wh/2.0))*c
box_right_cor=(box_xy+(box_wh/2.0))*c
with tf.Session() as sess:
                box_left=sess.run(box_left_cor)
                box_right=sess.run(box_right_cor)
                scores=sess.run(scores)
                classes=sess.run(classes)
                box_confidence=sess.run(box_confidence)
#converting left and right corners as total box parameters
box=np.zeros((box_left.shape[0],4))
for i in range(box_left.shape[0]):
    x=box_left[i,0:]
    y=box_right[i,0:]
    box[i]=(x[0],x[1],y[0],y[1])
#intersection over union
def iou(box1,box2):
    i1=max(box1[0],box2[0])
    i2=min(box1[2],box2[2])
    j1=max(box1[1],box2[1])
    j2=min(box1[3],box2[3])

    common_area=(i1-i2)*(j1-j2)

    box1_area=(box1[3] - box1[1])*(box1[2]- box1[0])
    box2_area=(box2[3] - box2[1])*(box2[2]- box2[0])
    union_area=box1_area+box2_area-common_area
    iou=common_area/union_area
    #imp condition
    if i1==box2[0]:
        if i1>box1[2]:
            iou=0
    else:
        if i1>box2[2]:
            iou=0

    return iou
#implementing iou
p=np.zeros(box_left.shape[0])
for i in range(box_left.shape[0]):
    if p[i]==0:
        for j in range(i+1,box_left.shape[0]):
            if p[i]==0:
                
                io=iou(box[i],box[j])
                if io>=0.4:
                    
                    if scores[i]>scores[j]:
                        p[j]=1
                        
                    else:
                        p[i]=1
                        
                        
t=0
t1=np.count_nonzero(p)
b=np.zeros((box_left.shape[0]-t1,4))
s=np.zeros(box_left.shape[0]-t1)
c=np.zeros(box_left.shape[0]-t1)
bc=np.zeros(box_left.shape[0]-t1)
for i in range(box_left.shape[0]):
    
    if p[i]==0:
        b[t]=box[i]
        s[t]=scores[i]
        c[t]=classes[i]
        bc[t]=box_confidence[i]
        t=t+1
#done with iou
bo=box
box=b

hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.

#draw boxes
font=ImageFont.truetype("arial.ttf", 50)
draw = ImageDraw.Draw(image)
for i in range(box.shape[0]):

    draw.rectangle((box[i,0],box[i,1],box[i,2],box[i,3]),fill=None,outline=colors[int(c[i])],width=8)
    draw.text((box[i,0],box[i,1]),class_names[int(c[i])]+" "+str(int(bc[i]*100))+"%",fill=(0,200,0),font=font)
        
image.show()
