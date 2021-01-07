from torch.utils.data import Dataset
from settings import path_to_data, path_to_annotation


class Data(Dataset):
    def __init__(self):
        super(Data, self).__init__()

    def __len__(self):
        return 0

    def __getitem__(self, item):
        return 1
