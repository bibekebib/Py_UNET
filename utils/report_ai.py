import requests
import sys
import json
sys.path.append('../Image Segmentation')
from app import extract_attributes

values = 'outputs/report/output_values.json'
with open(values, 'r+') as f:
    value = json.load(f)

trn_loss = value['train_loss']
val_loss = value['val_loss']

# Append necessary paths
sys.path.append('Models')
sys.path.append('utils')

# Custom imports
from unet_model import UNET
from dice_loss import dice_loss
from misc import check_device, get_optimizer
data_channels = extract_attributes()[1]['n_channels']
optimizer_name = extract_attributes()[2]['optimizer']
learning_rate = extract_attributes()[2]['learning_rate']
train_batch_size=extract_attributes()[2]['train_batch_size'] # training batch size
test_batch_size=extract_attributes()[2]['test_batch_size']   
epochs = extract_attributes()[2]['epochs']

device = check_device()
model = UNET(data_channels)


# Provide the necessary information
prompt = f"""
- ImageSegmentation Model Using UNET
-  UNET Architecture which consists of three components: the ThreeConvBlock with triple convolution, batch normalization, and ReLU activation; the DownStage for downsampling with max-pooling and convolution; and the UpStage for upsampling, combining resized features, and using skip connections.
- dice loss
- Learning Rate: {learning_rate}
- Training Batch Size : {train_batch_size}
- Testing Batch Size: {test_batch_size}
- Epochs: {epochs}
- Train Loss: {trn_loss}
- Test Loss: {val_loss}
- forest image segmentation dataset with 5000+ image and mask.
Generate a comprehensive research paper expanding  section like Introduction, Background,  Literature Review,Methodology, conclusion and references  and ensuring a coherent structure with a minimum of 10000 words.
"""


def generate_research_paper(prompt, output_path):
    # Set up the API endpoint and headers
    url = "https://api.edenai.run/v2/text/generation"
    API_KEY = extract_attributes()[4]['edenai_api_key']
    # API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMDNiNjFmNjYtNjBjYy00YmU4LTg3MWMtNGEzY2VjNDBkNjc5IiwidHlwZSI6ImZyb250X2FwaV90b2tlbiJ9.OQGj8o2j1tPXNtWu47ABLd0v3d2HS6Uhbc2ciOSSCYs"
    headers = {
        "Authorization": f"Bearer {API_KEY}",  # Replace with your API key
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    # Prepare the payload for the API request
    payload = {
        "providers": "openai",
        "text": prompt,
        "temperature": 0.3,
    }

    # Make the API request
    response = requests.post(url, json=payload, headers=headers)

    # Parse the response JSON
    result = json.loads(response.text)
    # Save the generated research paper to the specified output path
    with open(output_path, "w") as f:
        f.write(result['openai']['generated_text'])

    print(f"Report saved to {output_path}")


# Specify the output path for the research paper
output_path = "outputs/report/project_report.md"

# Call the function to generate and save the research paper


def report_main():
    generate_research_paper(prompt, output_path)


report_main()