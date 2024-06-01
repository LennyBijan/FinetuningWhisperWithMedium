from huggingface_hub import create_repo, upload_folder, HfApi

# User Inputs
YOUR_TOKEN = "your_token_here"  # Replace 'your_token_here' with your actual HF token
DATASET_NAME = "Datasetname"  # Replace 'Datasetname' with your actual dataset name
LOCAL_PATH = "path_to/Datasetname/data"  # Replace with the path to your local dataset directory

# Setup repository name
username = "your_username_here"  # Replace 'your_username_here' with your HF username
repo_name = f"{username}/{DATASET_NAME}"

# Create a new repository on the Hugging Face Hub
create_repo(repo_name, private=False, token=YOUR_DELIVERY_HF_TOKEN, repo_type="dataset")

# Upload the data directory contents to the root of your newly created repository
upload_folder(
    folder_path=LOCAL_PATH,
    path_in_repo="",  # uploads directly to the root of the repository
    repo_id=repo_name,
    token=YOUR_TOKEN
)

# Print the URL of the repository
print(f"Your dataset is uploaded. You can view it at: https://huggingface.co/datasets/{repo_name}")