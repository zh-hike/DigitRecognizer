import torch
import torch.nn as nn
from dataset.dataLoader import DL
from network import Model
from sklearn.metrics import accuracy_score
import pandas as pd


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._init_data()
        self._init_model()

    def _init_model(self):
        self.net = Model().to(self.device)
        self.cri = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.net.parameters(), weight_decay=4e-4)

    def _init_data(self):
        data = DL(self.args)
        self.traindl = data.traindl
        self.testdl = data.testdl
        self.valdl = data.valdl

    @torch.no_grad()
    def val(self):
        self.net.eval()
        batch_pred = []
        batch_real = []
        for batch, (inputs, targets) in enumerate(self.traindl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred = self.net(inputs)
            batch_pred = batch_pred + pred.argmax(dim=1).cpu().detach().numpy().tolist()
            batch_real = batch_real + targets.cpu().numpy().tolist()

        val_acc = accuracy_score(batch_pred, batch_real)

        self.net.train()
        return val_acc

    def train(self):

        patten = 'Iter %d/%d   [==============]   loss: %.4f   train_acc: %.5f   val_acc: %.5f'
        for epoch in range(self.args.epochs):
            batch_loss = 0
            batch_pred = []
            batch_real = []
            for batch, (inputs, targets) in enumerate(self.traindl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                pred = self.net(inputs)
                loss = self.cri(pred, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                batch_loss += loss.item()
                batch_pred = batch_pred + pred.argmax(dim=1).cpu().detach().numpy().tolist()
                batch_real = batch_real + targets.cpu().numpy().tolist()

            train_acc = accuracy_score(batch_pred, batch_real)
            val_acc = self.val()
            print(patten % (
                epoch,
                self.args.epochs,
                batch_loss,
                train_acc,
                val_acc,
            ))
        self.test()

    @torch.no_grad()
    def test(self):
        self.net.eval()
        all_pred = []
        for batch, (inputs) in enumerate(self.testdl):
            inputs = inputs.to(self.device)
            pred = self.net(inputs)
            all_pred = all_pred + pred.argmax(dim=1).detach().cpu().numpy().tolist()

        data = {'ImageId':list(range(1, 1+len(all_pred))),
                'Label': all_pred}
        df = pd.DataFrame()
        df['ImageId'] = data['ImageId']
        df['Label'] = data['Label']
        df.to_csv('results/pred.csv', index=False)


