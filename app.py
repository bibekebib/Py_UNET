import json

def load_config(filename):
    with open(filename, 'r') as json_file:
        config = json.load(json_file)
    return config

def extract_attributes(config=load_config(filename='config.json')):
    data_loader = config["DataLoader"]
    data_info = config["Data"]
    model_info = config["Model"]
    misc_info = config["Misc"]
    report_info = config["Report"]

    data_loader_attributes = {
        "csv": data_loader["csv"],
        "csv_path": data_loader["csv_path"],
        "csv_train_path": data_loader["csv_train_path"],
        "csv_test_path": data_loader["csv_test_path"],
        "image_folder": data_loader["Image_folder"],
        "image_folder_path": data_loader["image_folder_path"],
        "split": data_loader["Split"]
    }

    data_attributes = {
        "n_channels": data_info["NChannels"],
        "height": data_info["Height"],
        "width": data_info["Width"]
    }

    model_attributes = {
        "train_batch_size": model_info["Train_Batch_size"],
        "test_batch_size": model_info["Test_Batch_size"],
        "optimizer": model_info["Optimizer"],
        "learning_rate": model_info["learning_rate"],
        "epochs":model_info['epochs']
    }

    misc_attributes = {
        "generate_log": misc_info["Generate_log"],
        "generate_plot": misc_info["Generate_plot"],
        "generate_report": misc_info["Generate_report"],
        "report_path": misc_info["report_path"]
    }

    report_attributes = {
        "edenai_api_key": report_info["EdenAI_API_Key"],
        "output_folder": report_info["output_folder"]
    }

    return data_loader_attributes, data_attributes, model_attributes, misc_attributes, report_attributes





