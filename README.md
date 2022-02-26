# MTRnet

## Description
This repository is implementation of MTRnet for Binary Change Detection in high resolution sattlelite imagery.

Since, optical satellites reached the space and humanity starts capturing the surface of the earth, change detection research has been studied. 
Change detection is a task to identify the change of interest from multiple images which taken at the different time step. 
Recently, change detection is actively researched with semantic segmentation approach in supervised manner along with rapidly developing deep learning techniques. 

We propose combination of simple techniques to improve the performance of supervised change detection models. 
To enhance the ability to capturing changed regions, we modify the architecture of supervised change detection models to get multi-scaled tiled input, and giving morphological guidance to the mod els. 
Evaluation of our method is performed based on the Change Detection Dataset(CD) and LEVIR-CD Dataset. 
The experiment result shows that our method is effective to every candidate models. 

![figure1](https://github.com/sungonuni/MTRnet_change_detection/blob/main/figure1.png)

![figure2](https://github.com/sungonuni/MTRnet_change_detection/blob/main/figure2.png)

![figure3](https://github.com/sungonuni/MTRnet_change_detection/blob/main/figure3.png)


## Requirement

- Python 3.8

- Pytorch 1.8.1

- torchvision 0.9.0

## Dataset

- [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit) (Change Detection Dataset)
- [SCD](https://drive.google.com/file/d/1cAyKCjRiRKfTysX1OqtVs6F1zbEI0EGj/view?usp=sharing) (SECOND CD Dataset)

## Train from scratch
    
    python train.py

## Evaluate model performance

    python eval.py

## Test set visualization

    python visualization.py
   
## Demo

    python visualization.py
  
## compare data

    python compareDATA.py
