# import kagglehub

# Download latest version
# path = kagglehub.dataset_download("omkarmanohardalvi/lungs-disease-dataset-4-types")
import kaggle

kaggle.api.dataset_download_files(
    "andrewmvd/drive-digital-retinal-images-for-vessel-extraction", 
    path="/mnt/c/dataset", 
    unzip=True  # 自动解压
)

print("Success!")
# print("Path to dataset files:", path)
