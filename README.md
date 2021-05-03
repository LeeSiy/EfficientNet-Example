# EfficientNet-Example
Example code for simple EfficientNet

# Settings
- python 3.7.0
- tensorflow 2.2.0

# Prepare Data
- make sure your data folder structure...
- image/
  - label1/
    - img1.jpg
    - img2.jpg
    - ...
  - label2/
    - img1.jpg
    - img2.jpg
    - ...
  - ...
- run datasets/create_data.py file with your path.
- result will be in datasets/temp file.

# run train.py file
- change output layer's class number as your custom datasets
- example result

![Screenshot from 2021-05-03 15-06-55](https://user-images.githubusercontent.com/62841284/116846724-ed4cd380-ac23-11eb-8322-e1a0c558170e.png)
