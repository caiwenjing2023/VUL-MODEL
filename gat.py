import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class VulnerabilityDetector(nn.Module):
    def __init__(self, num_features, num_classes):
        super(VulnerabilityDetector, self).__init__()
        self.conv1 = GATConv(num_features, 16, heads=8, dropout=0.6)
        self.conv2 = GATConv(16 * 8, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def detect_vulnerability():
    # 创建一个简单的图结构，用于示例
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
    y = torch.tensor([0, 1, 0], dtype=torch.long)

    num_features = x.size(1)
    num_classes = int(torch.max(y)) + 1

    # 创建一个GAT模型
    model = VulnerabilityDetector(num_features, num_classes)

    # 定义损失函数和优化器
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 模型训练
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # 模型评估
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index)
        predicted_labels = torch.argmax(output, dim=1)

    # 检测节点分类准确性漏洞
    if torch.all(predicted_labels == y):
        print("节点分类准确性漏洞存在！")
    else:
        print("节点分类准确性漏洞不存在。")

detect_vulnerability()
