# Detecting Retina Damage From Optical Coherence Tomography (OCT) Images
Model for detecting retina damage from Optical Coherence Tomography (OCT) Images, using Transfer Learning on VGG16 CNN Model.
## Context
Retinal Optical Coherence Tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time (Swanson and Fujimoto, 2017).
## Installation
1. pip install -r requirements.txt
2. python app.py
3. Selectionner l'image OCT en format JPEG à analyser et cliquez sur View
4. La classe déterminée par le modèle s'affiche sous l'image

CNV = class 0
DME = class 1
DUREN = class 2
NORMAL = class 3 

## Data:
Data is avalaible here : https://data.mendeley.com/datasets/rscbjbr9sj/2
- Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

## Model 
A VGG16 CNN architecture is used for calssification pretrained on the 'ImageNet' dataset. 
The full code is available here https://colab.research.google.com/drive/1UzymPZ7DOG9JO2nOEA4IndMaed1kzQyK?usp=sharing
