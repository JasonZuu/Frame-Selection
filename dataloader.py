from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 ** kwargs):
        self.video_paths = data["video_path"].values
        self.labels = data["label"].values

    def __len__(self):
        return self.video_paths.shape[0]

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]
        return video_path, label


class MyDataLoader:
    def __init__(self,
                 data_csv_path: str,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 **kwargs):
        super().__init__()
        self.data = pd.read_csv(data_csv_path)[:100]
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = VideoDataset(self.data)

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )
    
    def data_count(self) -> int:
        return len(self.data)
