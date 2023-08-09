# Py_UNET
A UNET Architecture designed in PyTorch for image segmentation.

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)

## About <a name = "about"></a>

This is a project that allows users to implement UNET in PyTorch without having to husstle the design of the architecture. It allows users to simple give dataset path and run a `run.py` file. 

## Getting Started <a name = "getting_started"></a>

To get started simple clone this repo and install the requirements specified in the requirements.txt file.

```bash
git clone https://github.com/bibekebib/Py_UNET
cd Py_UNET
pip3 install requirements.txt
```
After that there is a `config.json` file where you simply changes all the requirements as per you need.
```
{
    "DataLoader":{
        "csv":"True", 
        "csv_path":"outputs/test.csv",
        "csv_train_path":"", 
        "csv_test_path":"",
        "Image_folder":"False",
        "image_folder_path":"",
        "Split":"True"
    },
    "Data":{
        "NChannels":3,
        "Height":256,
        "Width":256
    },
    "Model":{
        "Train_Batch_size":16,
        "Test_Batch_size":8,
        "Optimizer":"Adam",
        "learning_rate":1e-3,
        "epochs":5
    },
    "Misc":{
        "Generate_log":"True",
        "Generate_plot":"True",
        "Generate_report":"True",
        "report_path":"outputs/report/summary_log.md"
    },
    "Report":{
        "EdenAI_API_Key":"",
        "output_folder":"outputs/report/project_report.md"
    }
}
```

Make yourself a new `EdenAI_API_Key` at https://www.edenai.co/.





### Parameters in config.json file.


```
DataLoader:

csv: This flag indicates whether the data will be loaded from a CSV file. If set to "True," the data loader will use CSV files.
csv_path: The path to the main CSV file containing the data.
csv_train_path: Path to the CSV file for training data (if applicable).
csv_test_path: Path to the CSV file for testing data (if applicable).
Image_folder: This flag indicates whether the data will be loaded from image files. If set to "True," the data loader will use image files.
image_folder_path: The path to the folder containing the image data.
Split: This flag indicates whether the data will be split into training and testing sets. If set to "True," data splitting will occur.


Data:

NChannels: Number of color channels in the images (e.g., 3 for RGB).
Height: Height of the images in pixels.
Width: Width of the images in pixels.


Model:

Train_Batch_size: Batch size for training the model.
Test_Batch_size: Batch size for testing the model.
Optimizer: The optimizer algorithm to be used for model training (e.g., "Adam").
learning_rate: The learning rate used by the optimizer during training.
epochs: Number of epochs for which the model will be trained.


Misc:

Generate_log: This flag indicates whether a log will be generated during the process. If set to "True," a log will be created.
Generate_plot: This flag indicates whether plots will be generated (e.g., learning curves). If set to "True," plots will be created.
Generate_report: This flag indicates whether a report will be generated. If set to "True," a report will be created.
report_path: The path where the generated summary log will be saved.


Report:

EdenAI_API_Key: An API key for accessing EdenAI services.
output_folder: The folder where the project report will be saved in Markdown format.
```


After configuring `config.json` file, you can simple run `run.py` file and if your given path is legimitate, then it will start training. 

```
python3 run.py
```

### Let le give you a cool thing about this tool.
This generates a boiler plate for the research paper for you.
You can use it as your own boilerplate or just copy paste and modify according to your needs.
Just head over to `outputs/reports/project_report.md` and see a generated report paper just crafted by you.


### Supported devices
This supports all the devices, `CPU` or `CUDA` or `MPS`. It will automatically detect and do as per the host device configuration.



### Generate Model Log
It also generates a model summary log file to you. Check it out in `outputs/reports/summary_log.md`

### Generates Graphs
If you `Generate_plot=True`, you get a loss graph in the `outputs/graphs/` folder.

### Saved Model
The final model will be saved in the `outputs/model` folder as `model.pth` file.

### Prediction
You can do prediction over you custom data simple heading to `predict.py` file and giving path to your file.

### Contribution
If you liked this tool, you are always welcome to make you contribution. Simply fork it and raise your issues. 

#### I will try to make related changes over time. 
### Happy Coding
