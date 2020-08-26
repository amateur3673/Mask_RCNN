## Mask RCNN for VOC dataset

### 1.Usage

To train model on VOC dataset on pretrained coco dataset, run:

```
python3 voc.py train --train_data="path/to/train/images" --val_data="path/to/val/images" --weights=="coco"
```

To Inference, just run:

```
python3 voc.py inference --weights="path/to/weight_file" --image="path/to/detection/image" --name="image_name"
```

### 2. Result

Here's some result on Instance Segmentation

![](Images/image1.png)

![](Images/image2.png)

![](Images/image3.png)

![](Images/image4.png)

![](Images/image5.png)

![](Images/image6.png)

![](Images/image7.png)

![](Images/image8.png)

### 3. Evaluation

Run the ``evaluate.py`` to get the precision and recall:

```
python3 evaluate.py --weight_path="path/to/weight_file" --confidence=confidence --data_dir="path/to/dataset"
```

``confidence`` here is the confidence score, only accept prediction higher than ``confidence``.

Run ``result.py`` to get mAP.

0.82, only evaluate on car,dog,cat,person,horse class.

AP curve:

![](Images/APcurve.png)