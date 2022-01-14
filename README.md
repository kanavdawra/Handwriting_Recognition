
# HRecognizer

## Background and motivation

With the advent of machine learning digital optical character recognition (OCR) is mostly solved and not much research is required in solving that problem. But Handwriting recognition is still an ongoing research topic because of many challenges. My hope is that by using a rich dataset of handwritten words and using deep learning techniques I can contribute in some way

## Challanges

The challenge with this project was to get a rich enough and small enough dataset that can be trained on my limited hardware. Another challenge was to integrate the web app into my website without breaking functionality.

## Goals

Build a CRNN model which can successfully recognise handwriting with reasonable accuracy and ability to be integrated into any website

## Dataset

Dataset used in this project is the transcriptions of 400,000 handwritten names on Kaggle. This dataset contains 400k+ names since these are names that mean there are potential 400K handwritings.

## Deployment 

The project is deployed on my website [kanavdawra.com](https://kanavdawra.com/portfolio/hrecognizer/) using GCP Cloud Functions

## Limitations

Right now it can only predict a sentence that is 35 characters long but in future, there is potential for making it recognize a full-page article

## Run it 

1. Download the dataset from [kaggle](https://www.kaggle.com/landlord/handwriting-recognition) extract the contents in a folder and name it Data.
2. Clone the project file to a folder and name it Project.
3. Make sure both the Project and Data are on the same level.
4. Insatall all the dependencies and run the index.ipynb file in the Project folder  
