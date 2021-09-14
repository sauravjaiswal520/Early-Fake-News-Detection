
import torch
import torch.nn as nn
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# UPDATIONS 
class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor)->torch.Tensor:
        # x.shape: Batch x sequence length x input_size
        # h_0.shape:
        # output.shape: batch x sequence length x input_size
        output, h_n = self.rnn_encoder(x, h_0)

        # output.shape: batch x input_size
        output = output.mean(dim=1)
        return output


class CNNEncoder(nn.Module):
    def __init__(self, out_channels: int, kernel_size: tuple):
        super(CNNEncoder, self).__init__()
        self.cnn_encoder = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        # x.shape: batch x sequence length x kernel_size[1](input_size)
        # x.shape: batch x 1 x sequence length x kernel_size[1]
        x = x.unsqueeze(dim=1)
        # output.shape: batch x out_channels x sequence length - kernel_size[0] + 1
        output = F.relu(self.cnn_encoder(x))
        # output.shape: batch x out_channels
        output = output.mean(dim=2)
        return output


class DetectModel(nn.Module):
    def __init__(self, input_size,
                 hidden_size, rnn_layers,
                 out_channels, height, cnn_layers,
                 linear_hidden_size, linear_layers, output_size):
        super(DetectModel, self).__init__()
        self.rnn_encoder = RNNEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_layers)
        self.cnn_encoder = CNNEncoder(out_channels=out_channels, kernel_size=(height, input_size))

        self.linear = nn.Sequential(
            nn.Linear(hidden_size + out_channels, linear_hidden_size), nn.ReLU(inplace=True),
            *chain(*[(nn.Linear(linear_hidden_size, linear_hidden_size), nn.ReLU(inplace=True)) for i in range(linear_layers - 2)]),
            nn.Linear(linear_hidden_size, output_size)
        )

    def forward(self, x, h0):
        # h0 for rnn_encoder
        
        rnn_output = self.rnn_encoder(x, h0)
        cnn_output = self.cnn_encoder(x)
        cnn_output = cnn_output.squeeze()
       
        # output.shape: batch x (hidden_size + out_channels)
        # output.shape: batch x output_size
        output = torch.cat([rnn_output, cnn_output], dim=1)
        output = self.linear(output)

        return output

# Original Code
class NeuralNetwork(nn.Module):

    class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self):
        raise NotImplementedError

    def fit(self, X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred,
            X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev):

        if torch.cuda.is_available():
            self.cuda()

        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['reg'], amsgrad=True) #
        # self.optimizer = torch.optim.Adadelta(self.parameters(), weight_decay=self.config['reg'])

        X_train_source_wid = torch.LongTensor(X_train_source_wid)
        X_train_source_id = torch.LongTensor(X_train_source_id)
        X_train_user_id = torch.LongTensor(X_train_user_id)
        X_train_ruid = torch.LongTensor(X_train_ruid)
        y_train = torch.LongTensor(y_train)
        y_train_cred = torch.LongTensor(y_train_cred)
        y_train_rucred = torch.LongTensor(y_train_rucred)
 
        dataset = TensorDataset(X_train_source_wid, X_train_source_id, X_train_user_id, X_train_ruid, y_train, y_train_cred, y_train_rucred)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # UPDATE

        model1 = DetectModel(config['input_size'],
                      config['hidden_size'], config['rnn_layers'],
                      config['out_channels'], config['height'], config['cnn_layers'],
                      config['linear_hidden_size'], config['linear_layers'], config['output_size'])
        
        
        
        model1 = model1.to(self.device)
        
        self.optimizer1 = torch.optim.Adam(params= model1.parameters(), lr=config['lr'])
        self.criterion = nn.CrossEntropyLoss(ignore_index=3)
        self.initial_hidden_state = torch.zeros(config['rnn_layers'], config['batch_size'], config['hidden_size'], dtype=torch.float, requires_grad=False).to(self.device)

        
        loss_func = nn.CrossEntropyLoss()
        loss_func2 = nn.CrossEntropyLoss(ignore_index=3) # kya h 
        for epoch in range(1, self.config['epochs']+1):
            print("\nEpoch ", epoch, "/", self.config['epochs'])
            self.train()
            avg_loss = 0
            avg_acc = 0
            for i, data in enumerate(dataloader):
                with torch.no_grad():
                    X_source_wid, X_source_id, X_user_id, X_ruid, batch_y, batch_y_cred, batch_y_rucred = (item.cuda(device=self.device) for item in data)

                self.optimizer.zero_grad()
                logit, ulogit, rulogit, r = self.forward(X_source_wid, X_source_id, X_user_id, X_ruid)

                output = model1(r,  self.initial_hidden_state[:, :r.shape[0], :])
                loss3 = self.criterion(output,  batch_y)
                
                
                loss1 = loss_func(logit, batch_y)
                pub_loss = loss_func(ulogit, batch_y_cred)
                uloss = loss_func2(rulogit.view(-1, rulogit.size(-1)), batch_y_rucred.view(-1))
                loss = loss1 + pub_loss + uloss + loss3

                loss.backward()
                self.optimizer.step()

                corrects = (torch.max(logit, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
                accuracy = 100*corrects/len(batch_y)
                print('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(i, loss.item(), accuracy, corrects, batch_y.size(0)))

                avg_loss += loss.item()
                avg_acc += accuracy

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

            cnt = y_train.size(0) // batch_size + 1
            print("Average loss:{:.6f} average acc:{:.6f}%".format(avg_loss/cnt, avg_acc/cnt))
            if epoch > self.config['epochs']//2 and self.patience > 2: #
                print("Reload the best model...")
                self.load_state_dict(torch.load(self.config['save_path']))
                now_lr = self.adjust_learning_rate(self.optimizer)
                print(now_lr)
                self.patience = 0
            self.evaluate(X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, epoch)


    def adjust_learning_rate(self, optimizer, decay_rate=.5):
        now_lr = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            now_lr = param_group['lr']
        return now_lr


    def evaluate(self, X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid, y_dev, epoch):
        y_pred = self.predict(X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid)
        acc = accuracy_score(y_dev, y_pred)
        print("Val set acc:", acc)
        print("Best val set acc:", self.best_acc)

        if epoch >= self.config['epochs']//2 and acc > self.best_acc:  #
            self.best_acc = acc
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            print("save model!!!")
        else:
            self.patience += 1


    def predict(self, X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid):
        if torch.cuda.is_available():
            self.cuda()

        self.eval()
        y_pred = []
        X_dev_source_wid = torch.LongTensor(X_dev_source_wid)
        X_dev_source_id = torch.LongTensor(X_dev_source_id)
        X_dev_user_id = torch.LongTensor(X_dev_user_id)
        X_dev_ruid = torch.LongTensor(X_dev_ruid)

        dataset = TensorDataset(X_dev_source_wid, X_dev_source_id, X_dev_user_id, X_dev_ruid)
        dataloader = DataLoader(dataset, batch_size=32)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                X_source_wid, X_source_id, X_user_id, \
                X_ruid = (item.cuda(device=self.device) for item in data)

            logits, _, _ = self.forward(X_source_wid, X_source_id, X_user_id, X_ruid)
            predicted = torch.max(logits, dim=1)[1]
            y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred
