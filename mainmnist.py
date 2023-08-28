import torchvision.utils as vutils

import utils, torch, time, os, pickle
import matplotlib.pyplot as plt
from utils import  mixup_data,mixup_data2, mixup_criterion
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import dataloader
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                transforms.RandomAffine(0, (1/8,0))]) # max horizontal shift by 4

def nll_loss_neg(y_pred, y_true):
    out = torch.sum(y_true * y_pred, dim=1)
    return torch.mean(- torch.log((1 - out) + 1e-6))

def nll_loss_neg2(y_pred, y_true):
    out = torch.sum(y_true * y_pred, dim=1)
    return torch.mean(- torch.log(( out) + 1e-6))


l2loss = nn.MSELoss()


criterion = nn.CrossEntropyLoss()
mixup=0.48
def mixup_batch(mixup,real1,real2,fake):
                def one_batch():


                    data = torch.cat((real1,real2, fake))
                    ones1 = Variable(torch.ones(real1.size(0), 1))
                    ones2 = Variable(torch.ones(real2.size(0), 1))                    
                    zeros = Variable(torch.zeros(fake.size(0), 1))
                    perm = torch.randperm(data.size(0)).view(-1).long()
                    if True:
                        ones1 = ones1.cuda()
                        ones2 = ones2.cuda()

                        zeros = zeros.cuda()
                        perm = perm.cuda()
                    labels = torch.cat((ones1,ones2, zeros))
                    return data[perm], labels[perm]

                d1, l1 = one_batch()
                if mixup == 0:
                    return d1, l1
                d2, l2 = one_batch()
                alpha = Variable(torch.tensor(np.random.beta(mixup, mixup)) )

                #print(alpha)

                if True:
                    alpha = alpha.cuda()
                d = alpha * d1 + (1. - alpha) * d2
                l = alpha * l1 + (1. - alpha) * l2
                return d, l


def mixup_batch2(mixup,f,y):
                def one_batch():


                    data = f
                    perm = torch.randperm(data.size(0)).view(-1).long()
                    if True:
                        perm = perm.cuda()
                    labels =y 
                    return data[perm], labels[perm]

                d1, l1 = one_batch()
                if mixup == 0:
                    return d1, l1
                d2, l2 = one_batch()
                alpha = Variable(torch.tensor(np.random.beta(mixup, mixup)) )

                #print(alpha)

                if True:
                    alpha = alpha.cuda()
                d = alpha * d1 + (1. - alpha) * d2
                l = alpha * l1 + (1. - alpha) * l2
                return d, l
class generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        utils.initialize_weights(self)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x), F.log_softmax(x, dim=1)


class MarginGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = True
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 62
        self.num_labels = args.num_labels
        self.index = args.index
        self.lrC = args.lrC
        # load dataset
        self.labeled_loader, self.unlabeled_loader, self.test_loader = dataloader(self.dataset, self.input_size, self.batch_size, self.num_labels)
        data = self.labeled_loader.__iter__().__next__()[0]
        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.C = classifier()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.C_optimizer = optim.SGD(self.C.parameters(), lr=args.lrC, momentum=args.momentum)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.C.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        utils.print_network(self.C)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()
        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['C_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['C_real_loss']=[]
        self.train_hist['test_loss']=[]
        self.train_hist['correct_rate']=[]
        self.train_hist['exit']=[]
        self.best_acc = 0
        self.best_time = 0

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            cla=0
            dloss=0
            gloss=0
            self.G.train()
            epoch_start_time = time.time()

            if epoch == 0:
                correct_rate = 0
                while True:
                    cr=0
                    for iter, (x_, y_) in enumerate(self.labeled_loader):
                        if self.gpu_mode:
                            x_, y_ = x_.cuda(), y_.cuda()
                        self.C.train()
                        self.C_optimizer.zero_grad()
                        _, C_real = self.C(x_)
                        C_real_loss = F.nll_loss(C_real, y_)
                        C_real_loss.backward()
                        self.C_optimizer.step()
                        cr+=C_real_loss.item()


                        if iter == self.labeled_loader.dataset.__len__() // self.batch_size:
                            self.C.eval()
                            test_loss = 0
                            correct = 0
                            with torch.no_grad():
                                for data, target in self.test_loader:
                                    data, target = data.cuda(), target.cuda()
                                    _, output = self.C(data)
                                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                                    correct += pred.eq(target.view_as(pred)).sum().item()
                            test_loss /= len(self.test_loader.dataset)

                            print('\niter: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                                (iter), test_loss, correct, len(self.test_loader.dataset),
                                100. * correct / len(self.test_loader.dataset)
                                ))
                            correct_rate = correct / len(self.test_loader.dataset)
                            
                    cr /=self.labeled_loader.dataset.__len__()
                    self.train_hist['test_loss'].append(test_loss)
                    self.train_hist['C_real_loss'].append(cr)
                    self.train_hist['correct_rate'].append(correct_rate)
                    
                    gate = 0.8
                    if self.num_labels == 600:
                        gate = 0.93
                    elif self.num_labels == 1000:
                        gate = 0.95
                    elif self.num_labels == 3000:
                        gate = 0.97
                    if correct_rate > gate:
                        self.train_hist['exit'].append(len(self.train_hist['correct_rate']))
                        break

            correct_wei = 0
            number = 0
            labeled_iter = self.labeled_loader.__iter__()
            # print(self.labeled_loader.dataset.__len__())
            for iter, (x_u, y_u) in enumerate(self.unlabeled_loader):
                self.C.train()
                if iter == self.unlabeled_loader.dataset.__len__() // self.batch_size:
                    if epoch > 0:
                        print('\nPseudo tag: Accuracy: {}/{} ({:.0f}%)\n'.format(
                            correct_wei, number,
                            100. * correct_wei / number))
                    break

                try:
                    x_l, y_l = labeled_iter.__next__()
                except StopIteration:
                    labeled_iter = self.labeled_loader.__iter__()
                    x_l, y_l = labeled_iter.__next__()

                z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_l, y_l, x_u, y_u, z_ = \
                        x_l.cuda(), y_l.cuda(), x_u.cuda(), y_u.cuda(), z_.cuda()

                # update C network
                self.C_optimizer.zero_grad()

                _, C_labeled_pred = self.C(x_l)
                C_labeled_loss = F.nll_loss(C_labeled_pred, y_l)


                



                C_unlabeled_pred, _ = self.C(x_u)
                C_unlabeled_wei = torch.max(C_unlabeled_pred, 1)[1]
                correct_wei += C_unlabeled_wei.eq(y_u).sum().item()
                number += len(y_u)
                C_unlabeled_wei=C_unlabeled_wei.view(-1, 1)
                mostLikelyProbs = np.asarray([C_unlabeled_pred[i, C_unlabeled_wei[i]].item() for  i in range(len(C_unlabeled_pred))])
                confidenceThresh = 2
                toKeep = (mostLikelyProbs >= confidenceThresh) 
                toKeepn = (mostLikelyProbs < confidenceThresh)                 
                C_unlabeled_loss=0
                C_unlabeled_lossh=0
                C_unlabeled_lossl=0
                if sum(toKeep) != 0:
                    #print(sum(toKeep))
                    #k2+=sum(toKeep)
                    C_unlabeled_wei1 = torch.zeros(sum(toKeep), 10).cuda().scatter_(1, C_unlabeled_wei[toKeep], 1)
                    C_unlabeled_lossh = nll_loss_neg2(C_unlabeled_wei1, C_unlabeled_pred[toKeep])
                if sum(toKeepn) != 0:
                    C_unlabeled_wei2 = torch.zeros(sum(~toKeep), 10).cuda().scatter_(1, C_unlabeled_wei[~toKeep], 1)
                    C_unlabeled_lossl = nll_loss_neg2(C_unlabeled_wei2, C_unlabeled_pred[~toKeep])
                C_unlabeled_loss=C_unlabeled_lossl
                    #C_unlabeled_lossh.backward(retain_graph=True)








                #L_unlabeled_predtC = l2loss(C_unlabeled_pred1 , C_unlabeled_predt )

               

                C_fake_pred, _ = self.C(x_u)
                C_fake_wei = torch.max(C_fake_pred, 1)[1]                              
                C_fake_wei = C_fake_wei.view(-1, 1)
                C_fake_wei = torch.zeros(self.batch_size, 10).cuda().scatter_(1, C_fake_wei, 1)
                C_fake_loss = nll_loss_neg(C_fake_wei, C_fake_pred)
                d2,l2 = mixup_batch2(mixup,x_u,C_fake_wei)
                outputsm , _ = self.C(d2)
                outputsm = torch.max(outputsm, 1)[1]
                outputsm = outputsm.view(-1, 1)
                outputsm = torch.zeros(outputsm.size(0), 10).cuda().scatter_(1, outputsm, 1)               
                C_u_lossmix = nll_loss_neg2(outputsm, l2)
















                G_ = self.G(z_)
                C_fake_pred, _ = self.C(G_)
                C_fake_wei = torch.max(C_fake_pred, 1)[1]                              
                C_fake_wei = C_fake_wei.view(-1, 1)
                C_fake_wei = torch.zeros(self.batch_size, 10).cuda().scatter_(1, C_fake_wei, 1)
                C_fake_loss = nll_loss_neg(C_fake_wei, C_fake_pred)
                d2,l2 = mixup_batch2(mixup,G_,C_fake_wei)
                outputsm , _ = self.C(d2)
                outputsm = torch.max(outputsm, 1)[1]
                outputsm = outputsm.view(-1, 1)
                outputsm = torch.zeros(outputsm.size(0), 10).cuda().scatter_(1, outputsm, 1)               
                C_fake_lossmix = nll_loss_neg(outputsm, l2)
                #print(C_unlabeled_loss)
                #print(C_fake_lossmix)
                #print(C_fake_loss)
                #print(C_u_lossmix)

                C_loss = C_labeled_loss  +0.1*C_fake_lossmix+ 0.1*C_fake_loss+0.1*C_u_lossmix+ C_unlabeled_loss
                self.train_hist['C_loss'].append(C_loss.item())

                C_loss.backward()
                self.C_optimizer.step()

                # update D network
                
                self.D_optimizer.zero_grad()


                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                D_labeled = self.D(x_l)
                D_labeled_loss = self.BCE_loss(D_labeled, torch.ones_like(D_labeled))
                D_unlabeled = self.D(x_u)
                D_unlabeled_loss = self.BCE_loss(D_unlabeled, torch.ones_like(D_unlabeled))                
                
                d2,l2 = mixup_batch(mixup,x_l,x_u,G_)
                
                D_loss_mix = self.BCE_loss(self.D(d2), l2)
                #D_loss=D_loss_mix
                D_loss = D_loss_mix#+D_labeled_loss + D_unlabeled_loss + D_fake_loss
                #self.train_hist['D_loss'].append(D_loss.item())
                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()




                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss_D = self.BCE_loss(D_fake, self.y_fake_)
                D_labeled = self.D(x_l)
                D_labeled_loss = self.BCE_loss(D_labeled, torch.ones_like(D_labeled))
                D_unlabeled = self.D(x_u)
                D_unlabeled_loss = self.BCE_loss(D_unlabeled, torch.ones_like(D_unlabeled))                
                d,l = mixup_batch(0,x_l,x_u,G_)         
                G_loss_mix=-self.BCE_loss(self.D(d), l)






                _, C_fake_pred = self.C(G_)
                C_fake_wei = torch.max(C_fake_pred, 1)[1]
                G_loss_C  = F.nll_loss(C_fake_pred, C_fake_wei)




                G_ = self.G(z_)
                C_fake_pred, _ = self.C(G_)
                C_fake_wei = torch.max(C_fake_pred, 1)[1]                         
                C_fake_wei = C_fake_wei.view(-1, 1)
                C_fake_wei = torch.zeros(self.batch_size, 10).cuda().scatter_(1, C_fake_wei, 1)
                d2,l2 = mixup_batch2(mixup,G_,C_fake_wei)
                outputsm , _ = self.C(d2)
                outputsm = torch.max(outputsm, 1)[1]
                outputsm = outputsm.view(-1, 1)
                outputsm = torch.zeros(outputsm.size(0), 10).cuda().scatter_(1, outputsm, 1)              
                G_C_fake_lossix = nll_loss_neg2(outputsm, l2)

                
                
                





                G_loss = G_loss_mix+0.1*G_C_fake_lossix+0.1*G_loss_C#+0.1*lossmcfg2#+G_loss_D

  












                
                #self.train_hist['G_loss'].append(G_loss.item())

                #G_loss_D.backward(retain_graph=True)
                #G_loss_C.backward()
                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()


                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, C_loss: %.8f" %
                          (
                          (epoch + 1), (iter + 1), self.unlabeled_loader.dataset.__len__() // self.batch_size, D_loss.item(),
                          G_loss.item(), C_loss.item()))
                cla+=C_loss.item()
                dloss+=D_loss.item()
                gloss+=G_loss.item()
                

            self.C.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.cuda(), target.cuda()
                    _, output = self.C(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.test_loader.dataset)

            acc = 100. * correct / len(self.test_loader.dataset)
            cur_time = time.time() - start_time
            with open('acc_time/LUG/' + str(self.num_labels) + '_' + str(self.index) + '_' + str(self.lrC) + '.txt', 'a') as f:
                f.write(str(cur_time) + ' ' + str(acc) + '\n')

            cla/=self.unlabeled_loader.dataset.__len__()
            dloss/=self.unlabeled_loader.dataset.__len__()
            gloss/=self.unlabeled_loader.dataset.__len__()
            self.train_hist['D_loss'].append(dloss)
            self.train_hist['G_loss'].append(gloss)
            self.train_hist['test_loss'].append(test_loss)
            self.train_hist['C_real_loss'].append(cla)
            self.train_hist['correct_rate'].append(acc)
            if acc > self.best_acc:
                self.best_acc = acc
                self.best_time = cur_time
                self.save()
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch + 1))
               
                vutils.save_image(d2,
                'c/inter_samples%s.png' % (epoch + 1),
                normalize=True)
                vutils.save_image(d2,
                'c/inter_samples%s.pdf' % (epoch + 1),
                normalize=True)       

                vutils.save_image(d,
                'dg/inter_samples%s.png' % (epoch + 1),
                normalize=True)
                vutils.save_image(d,
                'dg/inter_samples%s.pdf' % (epoch + 1),
                normalize=True)              

        with open('acc_time/LUG/' + str(self.num_labels) + '_' + str(self.lrC) + '_best.txt', 'a') as f:
            f.write(str(self.index) + ' ' + str(self.best_time) + ' ' + str(self.best_acc) + '\n')
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch,
                                                                        self.train_hist['total_time'][0]))
        print("Training finish!... save training results")





    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        path = self.result_dir + '/LUG/' + self.model_name + '_' + str(self.index)

        if not os.path.exists(path):
            os.makedirs(path)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

            
        samples1=samples


        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          path + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
        gridOfFakeImages = torchvision.utils.make_grid((samples1 + 1) / 2)
        torchvision.utils.save_image(gridOfFakeImages,path + '/' + self.model_name + '_epoch%03d' % epoch + '.pdf')

    def save(self):

        save_dir = self.save_dir + '/LUG/' + self.model_name + '_' + str(self.index)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
        torch.save(self.C.state_dict(), os.path.join(save_dir, self.model_name + '_C.pkl'))
        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)


    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        #self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        #self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
        self.C.load_state_dict(torch.load('/content/models/LUG/MarginGAN_1/MarginGAN_C.pkl'))
