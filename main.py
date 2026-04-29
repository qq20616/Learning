import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm  # ← 在导包区添加这行

# ==================== 参数配置说明 ====================
# DATA_PATH       : 数据集路径
# BATCH_SIZE      : 每批样本数
# EPOCHS          : 最大训练轮数（早停可能提前结束）
# LR              : 学习率
# EMBED_DIM       : 隐向量维度
# MLP_LAYERS      : MLP 隐藏层结构
# NUM_NEGATIVES   : 负采样比例
# WEIGHT_DECAY    : L2 正则化强度，建议 1e-4 ~ 1e-5，防止过拟合
# PATIENCE        : 早停容忍轮数，连续 N 轮 AUC 不提升则停止
# DEVICE          : GPU/CPU 自动选择
# =====================================================

DATA_PATH = r"D:\PythonProject\Movierecommendsystem\ml-1m"
BATCH_SIZE = 256
EPOCHS = 20              # 放宽到 20，让早停自己决定何时结束
LR = 0.001
EMBED_DIM = 64
MLP_LAYERS = [128, 64, 32]
NUM_NEGATIVES = 4
WEIGHT_DECAY = 1e-4    # ← 新增：L2 正则化
PATIENCE = 3           # ← 新增：早停容忍 3 轮
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 1. 数据预处理 ====================
def load_data(data_path):
    ratings = pd.read_csv(
        f"{data_path}/ratings.dat", sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"], encoding="latin-1"
    )
    movies = pd.read_csv(
        f"{data_path}/movies.dat", sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1"
    )

    # 重映射 ID 为连续索引（关键！防止 Embedding 越界）
    user_map = {id: i for i, id in enumerate(ratings["user_id"].unique())}
    movie_map = {id: i for i, id in enumerate(ratings["movie_id"].unique())}
    ratings["user_idx"] = ratings["user_id"].map(user_map)
    ratings["movie_idx"] = ratings["movie_id"].map(movie_map)

    num_users = len(user_map)
    num_movies = len(movie_map)
    return ratings, movies, user_map, movie_map, num_users, num_movies


def build_implicit_dataset(ratings, num_negatives=4):
    """构建隐式反馈数据：评分>=4为正样本，其余忽略，并负采样"""
    pos = ratings[ratings["rating"] >= 4].copy()
    user_watched = pos.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    all_movies = set(range(num_movies))

    train_data = []
    test_data = []

    for user_idx, group in pos.groupby("user_idx"):
        movies = group["movie_idx"].tolist()
        # 时序划分：最后一条作测试
        test_movie = movies[-1]
        train_movies = movies[:-1]

        # 测试集：1正 + 100负（用于简单评估）
        test_negs = np.random.choice(
            list(all_movies - user_watched.get(user_idx, set())),
            size=100, replace=False
        )
        test_data.append((user_idx, test_movie, 1))
        for n in test_negs:
            test_data.append((user_idx, n, 0))

        # 训练集：1正 + num_negatives负
        for m in train_movies:
            train_data.append((user_idx, m, 1))
            negs = np.random.choice(
                list(all_movies - user_watched.get(user_idx, set()) - {m}),
                size=num_negatives, replace=False
            )
            for n in negs:
                train_data.append((user_idx, n, 0))

    return train_data, test_data


class NCFDataset(Dataset):
    def __init__(self, data):
        self.users = torch.LongTensor([x[0] for x in data])
        self.movies = torch.LongTensor([x[1] for x in data])
        self.labels = torch.FloatTensor([x[2] for x in data])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]


# ==================== 2. 模型定义 ====================
class NeuMF(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim, mlp_layers, dropout=0.3):
        super(NeuMF, self).__init__()
        # GMF
        self.gmf_user = nn.Embedding(num_users, embed_dim)
        self.gmf_movie = nn.Embedding(num_movies, embed_dim)
        # MLP
        self.mlp_user = nn.Embedding(num_users, embed_dim)
        self.mlp_movie = nn.Embedding(num_movies, embed_dim)

        mlp_input = embed_dim * 2
        layers = []
        for out_dim in mlp_layers:
            layers += [nn.Linear(mlp_input, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            mlp_input = out_dim
        self.mlp = nn.Sequential(*layers)
        # 输出
        self.output = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user_idx, movie_idx):
        # GMF: 逐元素乘积
        gmf = self.gmf_user(user_idx) * self.gmf_movie(movie_idx)
        # MLP: 拼接后深度网络
        mlp = torch.cat([self.mlp_user(user_idx), self.mlp_movie(movie_idx)], dim=-1)
        mlp = self.mlp(mlp)
        # 融合
        vec = torch.cat([gmf, mlp], dim=-1)
        return self.sigmoid(self.output(vec).squeeze())


# ==================== 3. 训练与评估 ====================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for u, m, y in loader:
        u, m, y = u.to(device), m.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(u, m)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_auc(model, loader, device):
    """简化评估：计算 AUC（适合半天冲刺，比 HR@K 好写）"""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for u, m, y in loader:
            u, m = u.to(device), m.to(device)
            pred = model(u, m).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.numpy())
    return roc_auc_score(labels, preds)


# ==================== 4. 主流程 ====================
if __name__ == "__main__":
    print("加载数据...")
    ratings, movies, user_map, movie_map, num_users, num_movies = load_data(DATA_PATH)

    print("构建隐式反馈数据集...")
    train_data, test_data = build_implicit_dataset(ratings, NUM_NEGATIVES)
    train_loader = DataLoader(NCFDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(NCFDataset(test_data), batch_size=512, shuffle=False)

    print(f"用户数: {num_users}, 电影数: {num_movies}")
# 模型、损失函数、优化器初始化
    
    # 实例化 NeuMF，传入用户/电影数量、Embedding 维度、MLP 结构，并搬到 GPU/CPU
    model = NeuMF(num_users, num_movies, EMBED_DIM, MLP_LAYERS).to(DEVICE)
    # 二分类交叉熵损失 BCELoss，衡量预测概率与真实 0/1 标签的差距
    criterion = nn.BCELoss()
    # Adam 优化器，学习率 0.001，weight_decay=1e-4 是 L2 正则化，防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print("开始训练...")
    # ==================== 训练 + 早停 ====================
    best_auc = 0.0  # 记录历史最高 AUC
    best_epoch = 0  # 记录最优轮数
    counter = 0  # 连续未提升计数器

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        auc = eval_auc(model, test_loader, DEVICE)

        # 打印当前轮结果
        tqdm.write(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {loss:.4f} | Test AUC: {auc:.4f}")

        # 早停逻辑：如果当前 AUC 创新高，保存模型并重置计数器
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # 保存最优参数
            tqdm.write(f"  → 新的最佳 AUC: {best_auc:.4f}，模型已保存")
        else:
            counter += 1
            tqdm.write(f"  → AUC 未提升，连续 {counter}/{PATIENCE} 轮")
            if counter >= PATIENCE:
                tqdm.write(f"\n早停触发！连续 {PATIENCE} 轮未提升，停止训练。")
                break

    print(f"\n{'=' * 50}")
    print(f"训练结束")
    print(f"最优结果出现在第 {best_epoch} 轮")
    print(f"最佳 Test AUC: {best_auc:.4f}")
    print(f"{'=' * 50}")

    # 加载最优模型用于后续推荐
    model.load_state_dict(torch.load("best_model.pth"))

    # ==================== 5. 推荐演示 ====================
    def recommend(user_id, top_k=10):
        user_idx = user_map.get(user_id)
        if user_idx is None:
            return "新用户（需冷启动处理）"

        # 候选：所有电影
        movie_indices = torch.arange(num_movies).to(DEVICE)
        user_tensor = torch.full((num_movies,), user_idx, dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            scores = model(user_tensor, movie_indices).cpu().numpy()

        # 过滤已观看
        watched = set(ratings[ratings["user_id"] == user_id]["movie_idx"].tolist())
        for m in watched:
            scores[m] = -1

        top_idx = np.argsort(scores)[::-1][:top_k]
        reverse_map = {v: k for k, v in movie_map.items()}

        print(f"\n为用户 {user_id} 推荐：")
        for i, idx in enumerate(top_idx, 1):
            mid = reverse_map[idx]
            title = movies[movies["movie_id"] == mid]["title"].values[0]
            print(f"{i}. {title} (score: {scores[idx]:.4f})")


    recommend(user_id=1)