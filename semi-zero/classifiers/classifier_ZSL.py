#author: akshitac8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import datasets.ZSLDataset as util
from sklearn.preprocessing import MinMaxScaler 
import torch.nn.functional as F
import copy

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True, netR=None, dec_size=4096, dec_hidden_size=4096,feature_type='vha'):
        self.train_X =  _train_X.clone() 
        self.train_Y = _train_Y.clone() 
        self.test_seen_feature = data_loader.test_seen_feature.clone()
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature.clone()
        self.test_unseen_label = data_loader.test_unseen_label
        self.test_unseen_mapped_label = data_loader.test_unseen_mapped_label
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model = LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.feature_type = feature_type
        self.netR = netR if self.feature_type != 'v' else None
        if self.netR:
            if self.feature_type == 'a':
                print('use only A !!!!!')
                self.netR.eval()
                self.input_dim = dec_size
            elif self.feature_type == 'h':
                self.netR.eval()
                print('use only H !!!!!')
                self.input_dim = dec_hidden_size
            else:
                print('use  VHA !!!!!')
                self.netR.eval()
                self.input_dim = self.input_dim + dec_size
                self.input_dim += dec_hidden_size
            self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
            self.train_X = self.compute_dec_out(self.train_X, self.input_dim)
            self.test_unseen_feature = self.compute_dec_out(self.test_unseen_feature, self.input_dim)
            self.test_seen_feature = self.compute_dec_out(self.test_seen_feature, self.input_dim)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        self.lr = _lr
        self.beta = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        if generalized:
            self.acc_seen, self.acc_unseen, self.H, self.epoch= self.fit()
            print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
        elif self.feature_type == 'a':
            unseen_feature = torch.tensor(data_loader.test_unseen_feature).cuda()
            syn_att = netR(unseen_feature)
            un_att = torch.tensor(data_loader.unseen_att).cuda()
            sim = syn_att @ un_att.transpose(0,-1)
            predict = torch.argmax(sim,dim=1).cpu()
            label = torch.tensor(data_loader.test_unseen_mapped_label)
            acc = torch.sum(predict==label) / len(label)


            # distance = syn_att.unsqueeze(1) - un_att.unsqueeze(0) # N, c, d
            # distance = distance.pow(2).sum(2).detach().cpu()
            # predict = torch.argmin(distance,dim=1)
            # label = torch.tensor(data_loader.test_unseen_mapped_label)
            # acc = torch.sum(predict==label) / len(label)
            self.acc = acc
        else:
            self.acc,self.best_model = self.fit_zsl() 
            #print('acc=%.4f' % (self.acc))
    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8 
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                #print('Training classifier loss= ', loss.data[0])
            acc,per_acc,frequency, pse_label, out= self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            print(f"val acc:{acc}")
            #print('acc %.4f' % (acc))
            if acc > best_acc:
                best_acc = acc
                best_per_acc = per_acc
                best_model = copy.deepcopy(self.model.state_dict())
        self.per_acc = per_acc
        self.frequency = frequency
        self.pse_label = pse_label
        self.out = out
        return best_acc, best_model 
        
    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        out = []
        best_model = copy.deepcopy(self.model.state_dict())
        # early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc_seen = 0
            acc_unseen = 0
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            print("S U H:",acc_seen,acc_unseen,H)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return best_seen, best_unseen, best_H,epoch
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]


    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda(), volatile=True)
            else:
                inputX = Variable(test_X[start:end], volatile=True)
            output = self.model(inputX)  
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class 

    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        logits = torch.FloatTensor(test_X.size(0), target_classes.size(0))
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda(), volatile=True)
            else:
                inputX = Variable(test_X[start:end], volatile=True)
            output = self.model(inputX)
            logit = self.model.get_logic()
            logits[start:end] = logit
            # frequency += output.detach().cpu().sum(0)
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        cls_center = []
        for la in range(len(self.unseenclasses)):
            center = test_X[torch.where(predicted_label == la)[0]].mean(0)
            if center.isnan().any():
                center = self.train_X[torch.where(self.train_Y == la)[0]].mean(0)
            cls_center.append(center)
        self.cls_center = torch.stack(cls_center).detach().cpu() 
        frequency = predicted_label.bincount()/len(test_label)
        frequency = frequency/frequency.sum()

        pse_label = predicted_label
        out = logits

        acc,per_acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc,per_acc,np.array(frequency), pse_label, out

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        return acc_per_class.mean() ,np.round(np.array(acc_per_class),2)

    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda(), volatile=True)
            else:
                inputX = Variable(test_X[start:end], volatile=True)
            feat1 = self.netR(inputX)
            feat2 = self.netR.getLayersOutDet()
            if self.feature_type=='a':
                new_test_X[start:end] = torch.cat([feat1],dim=1).data.cpu()
            elif self.feature_type=='h':
                new_test_X[start:end] = torch.cat([feat2],dim=1).data.cpu()
            else:
                assert self.feature_type == 'vha'
                new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            start = end
        return new_test_X

class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        self.hidden = self.fc(x)
        o = self.logic(self.hidden)
        self.o = o
        return o
    def get_logic(self):
        return F.softmax(self.hidden,dim=1)

