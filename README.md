# FocusVision

End-to-end salient object detection project using a custom CNN model built with PyTorch


## Project Overview

This project focuses on salient object detection using deep learning and convolutional neural networks built with PyTorch

The main objective is to identify and segment the most visually important object in an image by generating saliency masks

During the project, multiple CNN architectures were implemented and compared, including:
 Baseline CNN model
 Improved CNN model using BCE + IoU loss
 Advanced encoder-decoder CNN with Batch Normalization and skip connections

 ## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
- Git & GitHub

## Dataset

The project uses the ECSSD (Extended Complex Scene Saliency Dataset), which contains RGB images together with corresponding ground truth saliency masks

The dataset was used for:
- training
- validation
- segmentation evaluation

## Features

- Custom CNN architectures
- Encoder-decoder segmentation pipeline
- Data augmentation
- BCE + IoU loss
- IoU / Precision / Recall / F1 evaluation
- Prediction visualization
- Training and validation loss analysis
- Model comparison experiments


## Results

The advanced CNN architecture achieved significantly better segmentation performance compared to the baseline models

### Final Advanced Model Metrics

- IoU Score: 0.4206
- Precision: 0.5787
- Recall: 0.6102
- F1 Score: 0.5940

The advanced model produced sharper and more focused segmentation masks during evaluation

## Future Improvements

Possible future improvements include:
- training on larger datasets
- increasing training epochs
- experimenting with attention mechanisms
- using pretrained encoders
- implementing more advanced segmentation architectures


## Author

Olti Miftari