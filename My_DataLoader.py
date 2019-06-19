from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np

class MyData(Dataset):
    def __init__(self, data_type='train', upsampling=False):
        assert data_type in ('train', 'valid', 'test')
        # Please replace them with your own data
        if data_type == 'train':
            data = [np.random.rand(128, 10).astype(np.float32), np.random.rand(128, 20).astype(np.float32),
                    np.random.randint(2, size=(128,1))]
        elif data_type == 'valid':
            data = [np.random.rand(128, 10).astype(np.float32), np.random.rand(128, 20).astype(np.float32),
                    np.random.randint(2, size=(128,1))]
        else:
            data = [np.random.rand(128, 10).astype(np.float32), np.random.rand(128, 20).astype(np.float32),
                    np.random.randint(2, size=(128,1))]
        self.unmodifiable_features = data[0]
        self.modifiable_features = data[1]
        self.labels = data[2]
        n = self.labels.shape[0]

        self.data_size = n

    def __getitem__(self, index):
        unmodifiable_feature = self.unmodifiable_features[index]
        modifiable_feature = self.modifiable_features[index]
        label = self.labels[index]
        return unmodifiable_feature, [torch.from_numpy(np.array([ele])) for ele in modifiable_feature], label

    def __len__(self):
        return self.data_size

if __name__ == '__main__':
    data = MyData(data_type='test')
    data_loader = DataLoader(data, batch_size=6, shuffle=False, num_workers=1)
    for batch_idx, (un, ch, targets) in enumerate(iter(data_loader)):
        print(batch_idx)
        print(un.dtype)
        print(ch[0].shape)
        print(targets)
        break
