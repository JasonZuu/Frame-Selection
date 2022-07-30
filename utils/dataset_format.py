import os
import pandas as pd

def get_dataset_format(data_dir:str) -> pd.DataFrame:
    data = {"video_path":[],
            "label":[]}
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            label = 1 if dir_name == "fake" else 0
            for file in os.listdir(dir_path):
                if file[-3:] == "mp4":
                    video_path = os.path.join(dir_path, file)
                    data["video_path"].append(video_path)
                    data["label"].append(label)
        break
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    data_dir = "data/datasets/IF_video"
    save_csv_path = "data/datasets/IF_video/data.csv"
    df = get_dataset_format(data_dir)
    df.to_csv(save_csv_path, index=False)