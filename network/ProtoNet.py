import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloaders.ProtoNetVideoDataset import ProtoNetVideoDataset
from mypath import Path
from network.C3D_model import C3D
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset

class ProtoNet(nn.Module):
    def __init__(self, encoder, num_classes):
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes

    def forward(self, support_set, query_set):
        support_features = self.encoder(support_set)
        query_features = self.encoder(query_set)

        support_features = support_features.view(support_features.size(0), support_features.size(1), -1)
        query_features = query_features.view(query_features.size(0), query_features.size(1), -1)

        n_support = support_features.size(2)
        n_query = query_features.size(2)

        support_features = support_features.unsqueeze(2).expand(-1, -1, n_query, -1)
        query_features = query_features.unsqueeze(3).expand(-1, -1, -1, n_support)
        prototypes = support_embeddings.view(self.num_classes, -1, support_embeddings.size(-1)).mean(dim=1)
        distances = torch.cdist(query_embeddings, prototypes)
        return -distances
dataset = 'hmdb51' # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes = 7
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

num_epochs = 20
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = C3D(num_classes).to(device)
    model = ProtoNet(encoder, num_classes=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 准备你的小样本数据集，如torch.utils.data.Dataset
    # 训练和验证数据集，这里略过具体实现
    # train_loader = DataLoader(VideoDataset(dataset=dataset, split='train',clip_len=10), batch_size=8, shuffle=True, num_workers=0)
    # val_loader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=10), batch_size=8, num_workers=0)
    # test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=10), batch_size=8, num_workers=0)

    train_loader = ProtoNetVideoDataset(VideoDataset(dataset=dataset, split='train',clip_len=10))
    val_loader   = ProtoNetVideoDataset(VideoDataset(dataset=dataset, split='val',  clip_len=10))
    test_dataloader  = ProtoNetVideoDataset(VideoDataset(dataset=dataset, split='test', clip_len=10))

    for epoch in range(num_epochs):
        # 训练
        model.train()
        for batch in train_loader:
            support_set, query_set, query_targets = batch["support_set"].to(device), batch["query_set"].to(device), \
            batch["query_targets"].to(device)
            optimizer.zero_grad()
            logits = model(support_set, query_set)
            loss = F.cross_entropy(logits, query_targets)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                support_set, query_set, query_targets = batch["support_set"].to(device), batch["query_set"].to(device),



if __name__ == "__main__":
    main()