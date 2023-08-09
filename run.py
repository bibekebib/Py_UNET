from app import extract_attributes, load_config
from predict import main
from train import generate_graph
from evaluate import get_dataloader

import sys
sys.path.append('utils')
from report_ai import report_main

# Extract attributes
data_loader_attrs, data_attrs, model_attrs, misc_attrs, report_attrs = extract_attributes()

csv = data_loader_attrs["csv"]
csv_path = data_loader_attrs["csv_path"]
csv_train_path = data_loader_attrs["csv_train_path"]
csv_test_path = data_loader_attrs["csv_test_path"]
image_folder = data_loader_attrs["image_folder"]
image_folder_path = data_loader_attrs["image_folder_path"]
split = data_loader_attrs["split"]

# Access Data attributes
n_channels = data_attrs["n_channels"]
height = data_attrs["height"]
width = data_attrs["width"]

# Access Model attributes
train_batch_size = model_attrs["train_batch_size"]
test_batch_size = model_attrs["test_batch_size"]
optimizer = model_attrs["optimizer"]
learning_rate = model_attrs["learning_rate"]
epochs = model_attrs["epochs"]  # Note the capitalization here

# Access Misc attributes
generate_log = misc_attrs["generate_log"]
generate_plot = misc_attrs["generate_plot"]
generate_report = misc_attrs["generate_report"]
report_path = misc_attrs["report_path"]

# Access Report attributes
edenai_api_key = report_attrs["edenai_api_key"]
output_folder = report_attrs["output_folder"]

train_ldr, test_ldr = get_dataloader(image_folder_path=image_folder_path, csv=csv, csv_path=csv_path, csv_train_path=csv_train_path, csv_test_path=csv_test_path, image_folder=image_folder, split=split,Batch_Size=train_batch_size, Test_Batch_Size=test_batch_size)
generate_graph(epochs=epochs, train_ldr=train_ldr, test_ldr=test_ldr)
if generate_log:
    main()
    print('Reports Generated')
else:
    print('Reports Not Generated')

report_main()

main()




