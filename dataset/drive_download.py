# import kagglehub
# path = kagglehub.dataset_download("andrewmvd/drive-digital-retinal-images-for-vessel-extraction")
# print(path)

import kaggle

kaggle.api.dataset_download_files(
    "andrewmvd/drive-digital-retinal-images-for-vessel-extraction", 
    path="/mnt/c/dataset", 
    unzip=True  # 自动解压
)

print("Success!")