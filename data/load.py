import kagglehub

path = kagglehub.dataset_download(
    "ashery/chexpert"
)

print(path)