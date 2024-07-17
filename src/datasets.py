import requests

def download_class_mapping():
    url = "https://huggingface.co/spaces/CVPR/VizWiz-CLIP-VQA/raw/main/data/annotations/class_mapping.csv"
    response = requests.get(url)
    with open("class_mapping.csv", "wb") as f:
        f.write(response.content)
    print("class_mapping.csv has been downloaded.")

if __name__ == "__main__":
    download_class_mapping()
