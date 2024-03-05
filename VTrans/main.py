import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import math, copy, time
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import random
from torch import optim
from torch.autograd import Variable
import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, precision_score, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--cancer_type', '-ct', type=str, default='skcm')

# VAE
parser.add_argument('--input_size', '-is', type=int, default=1406)
parser.add_argument('--hidden_size_1', '-hs1', type=int, default=256)
parser.add_argument('--hidden_size_2', '-hs2', type=int, default=20)
parser.add_argument('--vae_dropout_rate', '-vdr', type=float, default=0.1)
parser.add_argument('--vae_learning_rate', '-vlr', type=float, default=1e-5)
parser.add_argument('--L1_regularization', '-l1', type=float, default=1e-4)
parser.add_argument('--number_epochs', '-ne', type=int, default=80)

# Transformer
parser.add_argument('--transformer_learning_rate', '-tlr', type=float, default=8.865e-05)
parser.add_argument('--num_heads', '-nh', type=int, default=2)
parser.add_argument('--hidden_size', 'hz', type=int, default=20)
parser.add_argument('--num_layer', '-nl', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=56)
parser.add_argument('--dropout_rate', 'dr', type=float, default=0.3)
parser.add_argument('--scheduler_factor', '-sf', type=float, default=0.1)
parser.add_argument('--scheduler_patience', '-sp', type=int, default=10)
parser.add_argument('--t_L1_regularization', '-tl1', type=float, default=5e-3)
parser.add_argument('--k_fold', type=int, default=5)
parser.add_argument('--t_number_epochs', '-tne', type=int, default=66)
parser.add_argument('--weight_decay', 'wd', type=float, default=0.05)

config = parser.parse_args()
#skcm(1), brca(0), blca(1), hnsc(0), kirc(0), luad(0), lusc(1), ov(0), stad(0),

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 设置随机数种子
#seed = 114514  # 可以选择任何整数作为种子
seed = config.seed
# 设置PyTorch的随机种子
torch.manual_seed(seed)
random.seed(seed)
# （可选）设置CUDA的随机种子，如果你使用GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 设置NumPy的随机种子
np.random.seed(seed)

scaler = StandardScaler()
#scaler = MinMaxScaler()

class reparametrize(nn.Module):
    def __init__(self):
        super(reparametrize, self).__init__()

    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.shape)
        epsilon = epsilon.to(device)
        return z_mean + (z_log_var / 2).exp() * epsilon


class VaeEncoder(nn.Module):
    def __init__(self):
        super(VaeEncoder, self).__init__()
        self.Dense = nn.Linear(config.input_size, config.hidden_size_1)
        self.z_mean = nn.Linear(config.hidden_size_1, config.hidden_size_2)
        self.z_log_var = nn.Linear(config.hidden_size_1, config.hidden_size_1)
        self.dropout = nn.Dropout(p=config.vae_dropout_rate)
        self.sample = reparametrize()

    def forward(self, x):
        o = torch.nn.functional.relu(self.Dense(x))
        o = self.dropout(o)
        z_mean = self.z_mean(o)
        z_log_var = self.z_log_var(o)
        o = self.sample(z_mean, z_log_var)
        return o, z_mean, z_log_var


class VaeDecoder(nn.Module):
    def __init__(self):
        super(VaeDecoder, self).__init__()
        self.Dense = nn.Linear(config.hidden_size_2, config.hidden_size_1)
        self.out = nn.Linear(config.hidden_size_1, config.input_size)
        self.dropout = nn.Dropout(p=config.vae_dropout_rate)
        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        o = nn.functional.relu(self.Dense(z))
        o = self.dropout(o)
        o = self.out(o)
        return self.relu(o)


class Vae(nn.Module):
    def __init__(self):
        super(Vae, self).__init__()
        self.best_val_loss = 999.999
        self.encoder = VaeEncoder()
        self.decoder = VaeDecoder()

    def forward(self, x):
        o, mean, var = self.encoder(x)
        return self.decoder(o), mean, var

#reconstruction_function = nn.MSELoss(reduction='sum')
reconstruction_function = nn.MSELoss(reduction='mean')

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

model = Vae()
model = model.to(device)
lamda_l1 = config.L1_regularization
best_val_loss = 999.999
optimizer = optim.Adam(model.parameters(), lr=config.vae_learning_rate)
def pre_train_model(num_epochs):
    print('VAE.....................................................')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        num_batches = len(data_set[0])
        for batch_idx, data in enumerate(data_set[0]):
            starttime = datetime.datetime.now()
            mic_data, _ = data
            mic_data = Variable(mic_data)
            mic_data = (mic_data.to(device) if torch.cuda.is_available() else mic_data)
            recon_batch, mu, logvar = model(mic_data)
            loss = loss_function(recon_batch, mic_data, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_loss += loss.data[0]
            train_loss += loss.item()
            regularization_loss = 0.0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
            train_loss += lamda_l1 * regularization_loss
            '''
            if batch_idx % 100 == 0:
                endtime = datetime.datetime.now()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} time:{:.2f}s'.format(
                    epoch,
                    batch_idx * len(mic_data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(mic_data),
                    (endtime-starttime).seconds))
            '''
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(data_set[0].dataset)))
        #print(len(train_loader.dataset))


        print('val')
        for batch_idx, data in enumerate(data_set[1]):
            model.eval()
            starttime = datetime.datetime.now()
            mic_data, _ = data
            mic_data = Variable(mic_data)
            mic_data = (mic_data.to(device) if torch.cuda.is_available() else mic_data)
            recon_batch, mu, logvar = model(mic_data)
            loss = loss_function(recon_batch, mic_data, mu, logvar)
            # train_loss += loss.data[0]
            val_loss += loss.item()
            if batch_idx % 100 == 0:
                '''
                endtime = datetime.datetime.now()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} time:{:.2f}s'.format(
                    epoch,
                    batch_idx * len(mic_data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(mic_data),
                    (endtime-starttime).seconds))
                '''
        print('====> Epoch: {} Average vloss: {:.4f}'.format(
            epoch, val_loss / len(data_set[1].dataset)))
        print('===========================================================')
class MyDataset(Dataset):
    def  __init__(self, features, labels, is_train=True):
        self.features = features.astype(float)
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels.astype(float))

    def __getitem__(self, index):
        self.sample_features = self.features[index]
        self.sample_label = self.labels[index]
        return self.sample_features, self.sample_label

    def __len__(self):
        return len(self.features)

def dataset(type):
    train_features = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/train_kfold.csv', header=0, index_col=False, usecols=lambda column: column not in ['ENTITY_STABLE_ID']).to_numpy()
    test_features = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/test_kfold.csv', header=0, index_col=False, usecols=lambda column: column not in ['ENTITY_STABLE_ID']).to_numpy()

    print('Origin:')
    print(train_features)


    train_features = np.transpose(train_features)
    test_features = np.transpose(test_features)

    print('Tr:')
    print(train_features)



    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)


    #train_features = pd.DataFrame(train_features)
    print('stand:')
    print(train_features)

    features = [train_features, test_features]

    train = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/train_kfold.csv', usecols=lambda column: column not in ['ENTITY_STABLE_ID'])
    test = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/test_kfold.csv', usecols=lambda column: column not in ['ENTITY_STABLE_ID'])

    train_id = np.array(train.columns).tolist()
    test_id = np.array(test.columns).tolist()

    y = pd.read_csv('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/' + type + '_tcga_pan_can_atlas_2018' + '/Test_not_sort_data_kfold.csv')
    label = y['Label']
    Id = y['ID']

    train_label = []
    test_label = []
    for trid in range(len(train_id)):
        for i in range(len(Id)):
            if train_id[trid] == Id[i]:
                train_label.append(label[i])

    for teid in range(len(test_id)):
        for i in range(len(Id)):
            if test_id[teid] == Id[i]:
                test_label.append(label[i])

    train_labels = np.array(train_label)
    test_labels = np.array(test_label)
    labels = [train_labels, test_labels]

    num_features = train_features.shape[0]
    print('train features shape', train_features.shape)
    print('train labels shape', train_labels.shape)
    print('test features shape', test_features.shape)
    print('test labels shape', test_labels.shape)
    print(features[0])
    print()

    trainset = MyDataset(features[0], labels[0])
    testset = MyDataset(features[1], labels[1])
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    return train_loader, test_loader, trainset, testset

save_path = 'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/model/model_parameters.pth'
model.load_state_dict(torch.load(save_path))


#skcm(1), brca(0), blca(1), hnsc(0), kirc(0), luad(0), lusc(1), ov(0), stad(0),
#kirc, skcm, lusc
type = config.cancer_type
data_set = dataset(type)
#pre_train_model(55)
vae_train = model(data_set[2].features.to(device))
vae_test = model(data_set[3].features.to(device))


add_train = vae_train[0].to(device) + data_set[2].features.to(device)
add_test = vae_test[0].to(device) + data_set[3].features.to(device)

add_train = add_train.cpu().detach().numpy()
add_test = add_test.cpu().detach().numpy()

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, drop_out, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(drop_out)

        pe = torch.zeros(max_seq_len, d_model) #5000, 256
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)#5000, 1, 256
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0), :].transpose(0, 1))
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Initial embedding for raw input
class SrcEmbed(nn.Module):
    def __init__(self, input_dim, d_model):
        super(SrcEmbed, self).__init__()
        self.w = nn.Linear(input_dim, d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.w(x))

class Encoder(nn.Module):
    def __init__(self, layer, N, d_model, dropout, num_features):
        super(Encoder, self).__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.fc = nn.Linear(d_model, 2)


    def forward(self, x, mask=None):
        print()
        x = self.src_embed(x)
        x = self.position_encode(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.fc(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def l1_regularizer(weight, lambda_l1):
    return lambda_l1 * torch.norm(weight, 1)

def main(num_features, d_model, drop_prob, d_ff, N, num_heads):
    c = copy.deepcopy
    attn = MultiHeadedAttention(num_heads, d_model, drop_prob)
    ff = PositionwiseFeedForward(d_model, d_ff, drop_prob)
    encoder_layer = EncoderLayer(d_model, c(attn), c(ff), drop_prob)
    encoder = Encoder(encoder_layer, N, d_model, drop_prob, num_features).cuda()
    return encoder


input_dim = 1406  # 输入词汇大小
learning_rate =  8.865e-05
num_heads = 2  # 注意力头的数量
d_model = 20
num_layers = 4
d_ff = 56
drop_out = 0.3
factor = 0.1
patience = 10  # thr=10

best_loss = 999.999
best_acc = 0
accuracies = []
val_size = []
val_loss_list = []
index = 999
lamda_l1 = config.t_L1_regularization #0.005
y_pred_list_true = []
score = []

K = config.k_fold  # 假设选择K=5折交叉验证
kfold = KFold(n_splits=K, shuffle=True, random_state=seed)
p = 1
attset_train_or = MyDataset(add_train, data_set[2].labels.cpu().detach().numpy())
attset_test = MyDataset(add_test, data_set[3].labels.cpu().detach().numpy())

for train_indices, val_indices in kfold.split(attset_train_or):

    X_train, y_train = attset_train_or[train_indices][0], attset_train_or[train_indices][1]
    X_val, y_val = attset_train_or[val_indices][0], attset_train_or[val_indices][1]
    val_size.append(len(X_val))
    attset_train = MyDataset(np.array(X_train), np.array(y_train))
    attset_val = MyDataset(np.array(X_val), np.array(y_val))

    vae_train_loader = DataLoader(attset_train, batch_size=64, shuffle=True)
    vae_val_loader = DataLoader(attset_val, batch_size=64, shuffle=False)

    model = main(config.input_size, config.hidden_size, config.dropout_rate, config.d_ff, config.num_layer, config.num_heads).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.transformer_learning_rate, weight_decay=config.weight_decay)
    # optimizer = torch.optim.SGD(attention.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    criterion.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience, verbose=False)

    for epoch in range(config.t_number_epochs):
        model.train()
        total_loss = 0.0
        total_accuracy = 0
        for data in vae_train_loader:
            x_data, y_data = data
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            outputs = model(x_data)
            outputs = outputs[0]# X_train 是你的输入数据
            regularization_loss = 0.0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
            loss = criterion(outputs, y_data.long())  # y_train 是对应的标签
            loss = loss + lamda_l1 * regularization_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            total_loss = total_loss + loss.item()
            accuracy = (outputs.argmax(1) == y_data).sum()
            total_accuracy = total_accuracy + accuracy
        print('train_loss:', total_loss / len(attset_train))
        print()
        print('acc:', total_accuracy / len(attset_train))

    '''
    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for data in vae_val_loader:
            x_data, y_data = data
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            outputs = model(x_data)
            outputs = outputs[0]
            va_loss = criterion(outputs, y_data.long())
            val_loss = val_loss + va_loss.item()
            accuracy = (outputs.argmax(1) == y_data).sum()
            val_acc = val_acc + accuracy
        val_loss_list.append(val_loss / len(attset_val))
        
        t = val_acc / len(attset_val)
        accuracies.append(t)
        if t > best_acc:
            index = p
            best_acc = t
            best_model = model
            torch.save(best_model.state_dict(), 'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/model/best_model_tr.pth')
        
        t = val_loss / len(attset_val)
        accuracies.append(t)
        if t < best_loss:
            index = p
            best_loss = t
            best_model = model
            torch.save(best_model.state_dict(), 'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/model/best_model_tr.pth')
        
    print('new trainning................................................................................................\n')
    '''
    vae_test_loader = DataLoader(attset_test, batch_size=64, shuffle=False)

    #model.load_state_dict(torch.load('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/model/best_model_tr.pth'))
    #attention = torch.load('D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/model/attention_parameters_kfold.pth')
    model.eval()
    total_correct = 0
    total_samples = 0
    y_pred_list = []
    y_pred_score = []
    with torch.no_grad():
        for data in vae_test_loader:
            x_data, y_data = data
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            y_pred = model(x_data)[0]
            total_correct += (y_pred.argmax(1) == y_data).sum()
            total_samples += y_data.size(0)
            y_pred_list.append(y_pred.argmax(1))
            y_pred_score.append(y_pred[:, 1])
    print(total_correct, total_samples)
    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy}')
    accuracies.append(accuracy)
    if accuracy > best_acc:
        score = y_pred_score
        y_pred_list_true = y_pred_list
        index = p
        best_acc = accuracy
        best_model = model
        torch.save(best_model.state_dict(), 'D:/TCGA/Data03_cBioPortal_tcga_microbiome_data/model/best_model_VTrans_test.pth')

    p += 1

print(accuracies)
print(val_size)
print(val_loss_list)
print(index)
print(best_acc)
print(y_pred_list_true)


t = []
for i in range(len(y_pred_list_true)):
    for j in range(len(y_pred_list_true[i])):
        f = y_pred_list_true[i][j].cpu()
        t.append(f)
t = np.array(t)
y = data_set[3].labels.long()

s = []
for i in range(len(score)):
    for j in range(len(score[i])):
        f = score[i][j].cpu()
        s.append(f)
s = np.array(s)

print("Accuracy: {:.4f}".format(accuracy_score(y, t)))
print("Precision: {:.4f}".format(precision_score(y, t)))
print("Recall: {:.4f}".format(recall_score(y, t)))
print("F1_Score: {:.4f}".format(f1_score(y, t)))
print("Auc: {:.4f}".format(roc_auc_score(y, s)))
print()



print()