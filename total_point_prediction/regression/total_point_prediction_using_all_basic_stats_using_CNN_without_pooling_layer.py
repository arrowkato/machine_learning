#------------------------------------------------------------------------------
# Total point of a NBA player based on from 18's PTS to 32's point is predicted
# Comments of this file are explanation of python command,
# so it is not important to understand them if you know Pythorch
# 畳み込みは、スタッツごとに畳み込む
# poolingしないほうが良さげな気がしたので、なしでやってみる
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
dat = pd.read_csv('C:/anaconda3/envs/pytlesson/data/legend_retired_players_all_stats.csv', skiprows=None, encoding='utf-8')

# ndarray形式への変換を忘れずにしておく
trainPoint = dat.loc[:, [
# G
'18th_G','19th_G','20th_G','21th_G','22th_G','23th_G','24th_G','25th_G','26th_G','27th_G','28th_G','29th_G','30th_G','31th_G','32th_G',
# GS
'18th_GS','19th_GS','20th_GS','21th_GS','22th_GS','23th_GS','24th_GS','25th_GS','26th_GS','27th_GS','28th_GS','29th_GS','30th_GS','31th_GS','32th_GS',
# MP
'18th_MP','19th_MP','20th_MP','21th_MP','22th_MP','23th_MP','24th_MP','25th_MP','26th_MP','27th_MP','28th_MP','29th_MP','30th_MP','31th_MP','32th_MP',
# FG
'18th_FG','19th_FG','20th_FG','21th_FG','22th_FG','23th_FG','24th_FG','25th_FG','26th_FG','27th_FG','28th_FG','29th_FG','30th_FG','31th_FG','32th_FG',
# FGA
'18th_FGA','19th_FGA','20th_FGA','21th_FGA','22th_FGA','23th_FGA','24th_FGA','25th_FGA','26th_FGA','27th_FGA','28th_FGA','29th_FGA','30th_FGA','31th_FGA','32th_FGA',
# 3P
'18th_3P','19th_3P','20th_3P','21th_3P','22th_3P','23th_3P','24th_3P','25th_3P','26th_3P','27th_3P','28th_3P','29th_3P','30th_3P','31th_3P','32th_3P',
# 3PA
'18th_3PA','19th_3PA','20th_3PA','21th_3PA','22th_3PA','23th_3PA','24th_3PA','25th_3PA','26th_3PA','27th_3PA','28th_3PA','29th_3PA','30th_3PA','31th_3PA','32th_3PA',
# 2P
'18th_2P','19th_2P','20th_2P','21th_2P','22th_2P','23th_2P','24th_2P','25th_2P','26th_2P','27th_2P','28th_2P','29th_2P','30th_2P','31th_2P','32th_2P',
# 2PA
'18th_2PA','19th_2PA','20th_2PA','21th_2PA','22th_2PA','23th_2PA','24th_2PA','25th_2PA','26th_2PA','27th_2PA','28th_2PA','29th_2PA','30th_2PA','31th_2PA','32th_2PA',
# FT
'18th_FT','19th_FT','20th_FT','21th_FT','22th_FT','23th_FT','24th_FT','25th_FT','26th_FT','27th_FT','28th_FT','29th_FT','30th_FT','31th_FT','32th_FT',
# FTA
'18th_FTA','19th_FTA','20th_FTA','21th_FTA','22th_FTA','23th_FTA','24th_FTA','25th_FTA','26th_FTA','27th_FTA','28th_FTA','29th_FTA','30th_FTA','31th_FTA','32th_FTA',
# ORB
'18th_ORB','19th_ORB','20th_ORB','21th_ORB','22th_ORB','23th_ORB','24th_ORB','25th_ORB','26th_ORB','27th_ORB','28th_ORB','29th_ORB','30th_ORB','31th_ORB','32th_ORB',
# DRB
'18th_DRB','19th_DRB','20th_DRB','21th_DRB','22th_DRB','23th_DRB','24th_DRB','25th_DRB','26th_DRB','27th_DRB','28th_DRB','29th_DRB','30th_DRB','31th_DRB','32th_DRB',
# TRB
'18th_TRB','19th_TRB','20th_TRB','21th_TRB','22th_TRB','23th_TRB','24th_TRB','25th_TRB','26th_TRB','27th_TRB','28th_TRB','29th_TRB','30th_TRB','31th_TRB','32th_TRB',
# AST
'18th_AST','19th_AST','20th_AST','21th_AST','22th_AST','23th_AST','24th_AST','25th_AST','26th_AST','27th_AST','28th_AST','29th_AST','30th_AST','31th_AST','32th_AST',
# STL
'18th_STL','19th_STL','20th_STL','21th_STL','22th_STL','23th_STL','24th_STL','25th_STL','26th_STL','27th_STL','28th_STL','29th_STL','30th_STL','31th_STL','32th_STL',
# BLK
'18th_BLK','19th_BLK','20th_BLK','21th_BLK','22th_BLK','23th_BLK','24th_BLK','25th_BLK','26th_BLK','27th_BLK','28th_BLK','29th_BLK','30th_BLK','31th_BLK','32th_BLK',
# TOV
'18th_TOV','19th_TOV','20th_TOV','21th_TOV','22th_TOV','23th_TOV','24th_TOV','25th_TOV','26th_TOV','27th_TOV','28th_TOV','29th_TOV','30th_TOV','31th_TOV','32th_TOV',
# PF
'18th_PF','19th_PF','20th_PF','21th_PF','22th_PF','23th_PF','24th_PF','25th_PF','26th_PF','27th_PF','28th_PF','29th_PF','30th_PF','31th_PF','32th_PF',
# PTS
'18th_PTS','19th_PTS','20th_PTS','21th_PTS','22th_PTS','23th_PTS','24th_PTS','25th_PTS','26th_PTS','27th_PTS','28th_PTS','29th_PTS','30th_PTS','31th_PTS','32th_PTS'
]].values





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


# 1次元の配列を15*20の2次元配列にリサイズ
X_train = X_train.reshape(len(X_train), 1, 15, 20)
X_test  = X_test.reshape( len(X_test), 1, 15, 20)



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
        # 入力チャネル、出力チャネル、カーネルサイズ 5←前２年、その年、後ろ２年の５年分見る
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,1), padding=(2,0)) # 6は適当
        # poolingなし
        self.fc1 = nn.Linear( 1800, 128)
        self.fc2 = nn.Linear(  128, 128)
        self.fc3 = nn.Linear(  128, 128)
        self.fc4 = nn.Linear(  128,   1)
    
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
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

#1000 21223.333984375
#2000 12.448011875152588
#3000 10.613061904907227
#4000 0.23486602306365967
#5000 5164.9644775390625
#6000 0.08468206226825714
#7000 16.083628177642822
#8000 0.06407420337200165
#9000 2.5471484661102295
#10000 2.6536113023757935

# レブロンの18歳から32歳までの得点
lebron_pts = np.array([[
0.,79.,80.,79.,78.,75.,81.,76.,79.,62.,76.,77.,69.,76.,74.,
0.,79.,80.,79.,78.,74.,81.,76.,79.,62.,76.,77.,69.,76.,74.,
0.,3122.,3388.,3361.,3190.,3027.,3054.,2966.,3063.,2326.,2877.,2902.,2493.,2709.,2794.,
0.,622.,795.,875.,772.,794.,789.,768.,758.,621.,765.,767.,624.,737.,736.,
0.,1492.,1684.,1823.,1621.,1642.,1613.,1528.,1485.,1169.,1354.,1353.,1279.,1416.,1344.,
0.,63.,108.,127.,99.,113.,132.,129.,92.,54.,103.,116.,120.,87.,124.,
0.,217.,308.,379.,310.,359.,384.,387.,279.,149.,254.,306.,339.,282.,342.,
0.,559.,687.,748.,673.,681.,657.,639.,666.,567.,662.,651.,504.,650.,612.,
0.,1275.,1376.,1444.,1311.,1283.,1229.,1141.,1206.,1020.,1100.,1047.,940.,1134.,1002.,
0.,347.,477.,601.,489.,549.,594.,593.,503.,387.,403.,439.,375.,359.,358.,
0.,460.,636.,814.,701.,771.,762.,773.,663.,502.,535.,585.,528.,491.,531.,
0.,99.,111.,75.,83.,133.,106.,71.,80.,94.,97.,81.,51.,111.,97.,
0.,333.,477.,481.,443.,459.,507.,483.,510.,398.,513.,452.,365.,454.,542.,
0.,432.,588.,556.,526.,592.,613.,554.,590.,492.,610.,533.,416.,565.,639.,
0.,465.,577.,521.,470.,539.,587.,651.,554.,387.,551.,488.,511.,514.,646.,
0.,130.,177.,123.,125.,138.,137.,125.,124.,115.,129.,121.,109.,104.,92.,
0.,58.,52.,66.,55.,81.,93.,77.,50.,50.,67.,26.,49.,49.,44.,
0.,273.,262.,260.,250.,255.,241.,261.,284.,213.,226.,270.,272.,249.,303.,
0.,149.,146.,181.,171.,165.,139.,119.,163.,96.,110.,126.,135.,143.,134.,
0.,1654.,2175.,2478.,2132.,2250.,2304.,2258.,2111.,1683.,2036.,2089.,1743.,1920.,1954.
]],)
lebron_pts = scaler.transform(lebron_pts)
lebron_pts = lebron_pts.reshape(1,1,15,20)
lebron_pts = Variable(torch.Tensor(lebron_pts))
model(lebron_pts)
# predict on my machine : 35265.0859



# Scottie Pippenの18歳から32歳までの得点
Scottie_Pippen_pts = np.array([[
0.,0.,0.,0.,79.,73.,82.,82.,82.,81.,72.,79.,77.,82.,44.,
0.,0.,0.,0.,0.,56.,82.,82.,82.,81.,72.,79.,77.,82.,44.,
0.,0.,0.,0.,1650.,2413.,3148.,3014.,3164.,3123.,2759.,3014.,2825.,3095.,1652.,
0.,0.,0.,0.,261.,413.,562.,600.,687.,628.,627.,634.,563.,648.,315.,
0.,0.,0.,0.,564.,867.,1150.,1153.,1359.,1327.,1278.,1320.,1216.,1366.,704.,
0.,0.,0.,0.,4.,21.,28.,21.,16.,22.,63.,109.,150.,156.,61.,
0.,0.,0.,0.,23.,77.,112.,68.,80.,93.,197.,316.,401.,424.,192.,
0.,0.,0.,0.,257.,392.,534.,579.,671.,606.,564.,525.,413.,492.,254.,
0.,0.,0.,0.,541.,790.,1038.,1085.,1279.,1234.,1081.,1004.,815.,942.,512.,
0.,0.,0.,0.,99.,201.,199.,240.,330.,232.,270.,315.,220.,204.,150.,
0.,0.,0.,0.,172.,301.,295.,340.,434.,350.,409.,440.,324.,291.,193.,
0.,0.,0.,0.,115.,138.,150.,163.,185.,203.,173.,175.,152.,160.,53.,
0.,0.,0.,0.,183.,307.,397.,432.,445.,418.,456.,464.,344.,371.,174.,
0.,0.,0.,0.,298.,445.,547.,595.,630.,621.,629.,639.,496.,531.,227.,
0.,0.,0.,0.,169.,256.,444.,511.,572.,507.,403.,409.,452.,467.,254.,
0.,0.,0.,0.,91.,139.,211.,193.,155.,173.,211.,232.,133.,154.,79.,
0.,0.,0.,0.,52.,61.,101.,93.,93.,73.,58.,89.,57.,45.,43.,
0.,0.,0.,0.,131.,199.,278.,232.,253.,246.,232.,271.,207.,214.,109.,
0.,0.,0.,0.,214.,261.,298.,270.,242.,219.,227.,238.,198.,213.,116.,
0.,0.,0.,0.,625.,1048.,1351.,1461.,1720.,1510.,1587.,1692.,1496.,1656.,841.
]],) 
Scottie_Pippen_pts = scaler.transform(Scottie_Pippen_pts)
Scottie_Pippen_pts = Scottie_Pippen_pts.reshape(1,1,15,20)
Scottie_Pippen_pts = Variable(torch.Tensor(Scottie_Pippen_pts))
model(Scottie_Pippen_pts)
# predict on the model:17925.3223
# true total point 18940




# コービー・ブライアントの18歳から32歳までの得点
Kobe_Bryant_PTS = np.array([[
71.,79.,50.,66.,68.,80.,82.,65.,66.,80.,77.,82.,82.,73.,82.,
6.,1.,50.,62.,68.,80.,82.,64.,66.,80.,77.,82.,82.,73.,82.,
1103.,2056.,1896.,2524.,2783.,3063.,3401.,2447.,2689.,3277.,3140.,3192.,2960.,2835.,2779.,
176.,391.,362.,554.,701.,749.,868.,516.,573.,978.,813.,775.,800.,716.,740.,
422.,913.,779.,1183.,1510.,1597.,1924.,1178.,1324.,2173.,1757.,1690.,1712.,1569.,1639.,
51.,75.,27.,46.,61.,33.,124.,71.,131.,180.,137.,150.,118.,99.,115.,
136.,220.,101.,144.,200.,132.,324.,217.,387.,518.,398.,415.,336.,301.,356.,
125.,316.,335.,508.,640.,716.,744.,445.,442.,798.,676.,625.,682.,617.,625.,
286.,693.,678.,1039.,1310.,1465.,1600.,961.,937.,1655.,1359.,1275.,1376.,1268.,1283.,
136.,363.,245.,331.,475.,488.,601.,454.,542.,696.,667.,623.,483.,439.,483.,
166.,457.,292.,403.,557.,589.,713.,533.,664.,819.,768.,742.,564.,541.,583.,
47.,79.,53.,108.,104.,112.,106.,103.,95.,71.,75.,94.,90.,78.,83.,
85.,163.,211.,308.,295.,329.,458.,256.,297.,354.,364.,423.,339.,313.,336.,
132.,242.,264.,416.,399.,441.,564.,359.,392.,425.,439.,517.,429.,391.,419.,
91.,199.,190.,323.,338.,438.,481.,330.,398.,360.,413.,441.,399.,365.,388.,
49.,74.,72.,106.,114.,118.,181.,112.,86.,147.,111.,151.,120.,113.,99.,
23.,40.,50.,62.,43.,35.,67.,28.,53.,30.,36.,40.,37.,20.,12.,
112.,157.,157.,182.,220.,223.,288.,171.,270.,250.,255.,257.,210.,233.,243.,
102.,180.,153.,220.,222.,228.,218.,176.,174.,233.,205.,227.,189.,187.,172.,
539.,1220.,996.,1485.,1938.,2019.,2461.,1557.,1819.,2832.,2430.,2323.,2201.,1970.,2078.
]],)
Kobe_Bryant_PTS = scaler.transform(Kobe_Bryant_PTS)
Kobe_Bryant_PTS = Kobe_Bryant_PTS.reshape(1,1,15,20)
Kobe_Bryant_PTS = Variable(torch.Tensor(Kobe_Bryant_PTS))
model(Kobe_Bryant_PTS)
# predict on the model：33644.7656
# true total point: 33643


# ジョン・ストックトンの18歳から32歳までの得点
John_Stockton_pts = np.array([[
0.,0.,0.,0.,82.,82.,82.,82.,82.,78.,82.,82.,82.,82.,82.,
0.,0.,0.,0.,5.,38.,2.,79.,82.,78.,82.,82.,82.,82.,82.,
0.,0.,0.,0.,1490.,1935.,1858.,2842.,3171.,2915.,3103.,3002.,2863.,2969.,2867.,
0.,0.,0.,0.,157.,228.,231.,454.,497.,472.,496.,453.,437.,458.,429.,
0.,0.,0.,0.,333.,466.,463.,791.,923.,918.,978.,939.,899.,868.,791.,
0.,0.,0.,0.,2.,2.,7.,24.,16.,47.,58.,83.,72.,48.,102.,
0.,0.,0.,0.,11.,15.,39.,67.,66.,113.,168.,204.,187.,149.,227.,
0.,0.,0.,0.,155.,226.,224.,430.,481.,425.,438.,370.,365.,410.,327.,
0.,0.,0.,0.,322.,451.,424.,724.,857.,805.,810.,735.,712.,719.,564.,
0.,0.,0.,0.,142.,172.,179.,272.,390.,354.,363.,308.,293.,272.,246.,
0.,0.,0.,0.,193.,205.,229.,324.,452.,432.,434.,366.,367.,338.,306.,
0.,0.,0.,0.,26.,33.,32.,54.,83.,57.,46.,68.,64.,72.,57.,
0.,0.,0.,0.,79.,146.,119.,183.,165.,149.,191.,202.,173.,186.,194.,
0.,0.,0.,0.,105.,179.,151.,237.,248.,206.,237.,270.,237.,258.,251.,
0.,0.,0.,0.,415.,610.,670.,1128.,1118.,1134.,1164.,1126.,987.,1031.,1011.,
0.,0.,0.,0.,109.,157.,177.,242.,263.,207.,234.,244.,199.,199.,194.,
0.,0.,0.,0.,11.,10.,14.,16.,14.,18.,16.,22.,21.,22.,22.,
0.,0.,0.,0.,150.,168.,164.,262.,308.,272.,298.,286.,266.,266.,267.,
0.,0.,0.,0.,203.,227.,224.,247.,241.,233.,233.,234.,224.,236.,215.,
0.,0.,0.,0.,458.,630.,648.,1204.,1400.,1345.,1413.,1297.,1239.,1236.,1206.
]],)
John_Stockton_pts = scaler.transform(John_Stockton_pts)
John_Stockton_pts = John_Stockton_pts.reshape(1,1,15,20)
John_Stockton_pts = Variable(torch.Tensor(John_Stockton_pts))
model(John_Stockton_pts)
# predict on the model： 21369.1367
# true total point 19711



# Dennis Rodmanの18歳から32歳までの得点
Dennis_Rodman_pts = np.array([[
0.,0.,0.,0.,0.,0.,0.,77.,82.,82.,82.,82.,82.,62.,79.,
0.,0.,0.,0.,0.,0.,0.,1.,32.,8.,43.,77.,80.,55.,51.,
0.,0.,0.,0.,0.,0.,0.,1155.,2147.,2208.,2377.,2747.,3301.,2410.,2989.,
0.,0.,0.,0.,0.,0.,0.,213.,398.,316.,288.,276.,342.,183.,156.,
0.,0.,0.,0.,0.,0.,0.,391.,709.,531.,496.,560.,635.,429.,292.,
0.,0.,0.,0.,0.,0.,0.,0.,5.,6.,1.,6.,32.,15.,5.,
0.,0.,0.,0.,0.,0.,0.,1.,17.,26.,9.,30.,101.,73.,24.,
0.,0.,0.,0.,0.,0.,0.,213.,393.,310.,287.,270.,310.,168.,151.,
0.,0.,0.,0.,0.,0.,0.,390.,692.,505.,487.,530.,534.,356.,268.,
0.,0.,0.,0.,0.,0.,0.,74.,152.,97.,142.,111.,84.,87.,53.,
0.,0.,0.,0.,0.,0.,0.,126.,284.,155.,217.,176.,140.,163.,102.,
0.,0.,0.,0.,0.,0.,0.,163.,318.,327.,336.,361.,523.,367.,453.,
0.,0.,0.,0.,0.,0.,0.,169.,397.,445.,456.,665.,1007.,765.,914.,
0.,0.,0.,0.,0.,0.,0.,332.,715.,772.,792.,1026.,1530.,1132.,1367.,
0.,0.,0.,0.,0.,0.,0.,56.,110.,99.,72.,85.,191.,102.,184.,
0.,0.,0.,0.,0.,0.,0.,38.,75.,55.,52.,65.,68.,48.,52.,
0.,0.,0.,0.,0.,0.,0.,48.,45.,76.,60.,55.,70.,45.,32.,
0.,0.,0.,0.,0.,0.,0.,93.,156.,126.,90.,94.,140.,103.,138.,
0.,0.,0.,0.,0.,0.,0.,166.,273.,292.,276.,281.,248.,201.,229.,
0.,0.,0.,0.,0.,0.,0.,500.,953.,735.,719.,669.,800.,468.,370.
]],)
Dennis_Rodman_pts = scaler.transform(Dennis_Rodman_pts)
Dennis_Rodman_pts = Dennis_Rodman_pts.reshape(1,1,15,20)
Dennis_Rodman_pts = Variable(torch.Tensor(Dennis_Rodman_pts))
model(Dennis_Rodman_pts)
# predict on my machine:9468.3975
# true total point 6683



