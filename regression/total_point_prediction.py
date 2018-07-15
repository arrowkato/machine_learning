#------------------------------------------------------------------------------
# Total point of a NBA player based on from 18's PTS to 32's point is predicted
# Comments of this file are explanation of python command,
# so it is not important to understand them if you know Pythorch
#------------------------------------------------------------------------------

# PyTorchライブラリの読み込み
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Pandasライブラリの読み込み
import pandas as pd

# Numpyライブラリの読み込み
import numpy as np

# Matplotlibライブラリの読み込み
from matplotlib import pyplot as plt
%matplotlib inline



# 18歳から32歳までの得点を学習して生涯得点数を予測する
dat = pd.read_csv('C:/anaconda3/envs/pytlesson/data/legend_retired_players.csv', skiprows=None, encoding='utf-8')

# ndarray形式への変換を忘れずにしておく
trainPoint = dat.loc[:, ['18th','19th','20th','21th','22th','23th','24th','25th','26th','27th','28th','29th','30th','31th','32th']].values
point = dat.loc[:, ['total_point']].values

X_train, X_test, y_train, y_test = train_test_split(trainPoint, point, test_size=0.33, random_state=5)



# dtypeをintからfloatに変更しておく
X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test  = y_test.astype(np.float32)



# データの標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# torch.Tensorに変換
X_train = torch.from_numpy(X_train)
X_test  = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test  = torch.from_numpy(y_test)



# 説明変数と目的変数のテンソルをまとめる
train = TensorDataset(X_train, y_train)


# ミニバッチに分ける
train_loader = DataLoader(train, batch_size=100, shuffle=True)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear( 15, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128,   1) 
    
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# インスタンスの生成
model = Net()


# 誤差関数のセット
criterion = nn.MSELoss()

# 最適化関数のセット
optimizer = optim.Adam(model.parameters(), lr=0.001)



# 学習開始
for epoch in range(10000):
    total_loss = 0
    # 分割したデータの取り出し
    for train_x, train_y in train_loader:
        # 計算グラフの構築
        train_x, train_y = Variable(train_x), Variable(train_y)
        # 勾配をリセットする
        optimizer.zero_grad()
        # 順伝播の計算
        output = model(train_x)
        # 誤差の計算
        loss = criterion(output, train_y)
        # 逆伝播の計算
        loss.backward()
        # 重みの更新
        optimizer.step()
        # 誤差の累積
        total_loss += loss.data[0]
    # 累積誤差を100回ごとに表示
    if (epoch+1) % 1000 == 0:
        print(epoch+1, total_loss)




# レブロンの18歳から32歳までの得点
lebron_pts = torch.Tensor([[0.,1654., 2175., 2478., 2132., 2250., 2304., 2258., 2111., 1683., 2036., 2089., 1743., 1920., 1954.]],) 
lebron_pts = Variable(torch.Tensor(scaler.transform(lebron_pts)))
model(lebron_pts)
# predict on my machine 44789.4336



Kareem_Abdul_Jabbar_pts = torch.Tensor([[0., 0., 0., 0., 2361., 2596., 2822., 2292., 2191., 1949., 2275., 2152., 1600., 1903., 2034 ]],) 
Kareem_Abdul_Jabbar_pts  = Variable(torch.Tensor(scaler.transform(Kareem_Abdul_Jabbar_pts)))
model(Kareem_Abdul_Jabbar_pts)
# predict on my machine：32641.5020
# Kareem's total point 38387　: Note that Kareem's data is not in training data.





