import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import sys
from MTNEDataSet import MTNEDataSet

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 1


class MTNE_a:
    def __init__(self, file_path, emb_size=64, neg_size=10, hist_len=27, directed=False,
                 learning_rate=0.003, batch_size=1000, save_step=50, epoch_num=1):
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num

        self.data = MTNEDataSet(file_path, neg_size, hist_len, directed)
        self.node_dim = self.data.get_node_dim()

        if torch.cuda.is_available() and 0:
            with torch.cuda.device(DID):
                self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                    FType).cuda(), requires_grad=True)

                self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.tri_scores = Variable((torch.zeros(8) + 1.).type(FType).cuda(), requires_grad=True)
                self.att_param = Variable(torch.diag(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(emb_size), 1. / np.sqrt(emb_size), (emb_size,))).type(
                    FType).cuda()), requires_grad=True)
        else:
            self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                FType), requires_grad=True)

            self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)
            self.tri_scores = Variable((torch.zeros(8) + 1.).type(FType), requires_grad=True)
            self.att_param = Variable(torch.diag(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(emb_size), 1. / np.sqrt(emb_size), (emb_size,))).type(
                FType)), requires_grad=True)

        self.opt = SGD(lr=learning_rate, params=[self.node_emb, self.att_param, self.delta])
        self.loss = torch.FloatTensor()

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask, h_ab, h_cb, h_p, nh_nodes, nh_times, nh_time_mask, nh_ab, nh_cb, nh_p):  
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)

        tri_delta = softmax(self.tri_scores, dim = 0)

        tris = h_ab + h_cb*2 + h_p*4
        tri_d = tri_delta.index_select(0, Variable(tris.view(-1))).view(batch, self.hist_len)		
                
        p_mu = ((s_node_emb - t_node_emb)**2).sum(dim=1).neg()
        p_alpha = (((h_node_emb - t_node_emb.unsqueeze(1))**2).sum(dim=2).neg()) + (((h_node_emb - s_node_emb.unsqueeze(1))**2).sum(dim=2).neg())
        #att = softmax(p_alpha, dim = 1)
        delta_s = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        delta_t = self.delta.index_select(0, Variable(t_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        p_lambda = p_mu + (tri_d * p_alpha * torch.exp(delta_s * delta_t * Variable(d_time)) * Variable(h_time_mask)).sum(dim=1)
        
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        nh_node_emb = self.node_emb.index_select(0, Variable(nh_nodes.view(-1))).view(batch, self.neg_size, self.hist_len, -1)

        n_tris = nh_ab + 2*nh_cb + nh_p*4
        n_tri_d = tri_delta.index_select(0, Variable(n_tris.view(-1))).view(batch, self.neg_size, self.hist_len)

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb)**2).sum(dim=2).neg()
        n_alpha = (((nh_node_emb - n_node_emb.unsqueeze(2))**2).sum(dim = 3).neg()) + (((nh_node_emb - s_node_emb.unsqueeze(1).unsqueeze(2))**2).sum(dim = 3).neg())
        
        #n_att = softmax(n_alpha, dim = 2)
        delta_n = self.delta.index_select(0, Variable(n_nodes.view(-1))).view(batch, -1).unsqueeze(2)

        n_d_time = torch.abs(t_times.unsqueeze(1).unsqueeze(2) - nh_times) # (batch, neg_size, hist_len)
        #print(n_mu.size())
        #print(n_att.size())
        #print(n_alpha.size())
        #print(delta_n.size())
        #print(n_d_time.size())
        n_lambda = n_mu + (n_tri_d * n_alpha * torch.exp(delta_s.unsqueeze(2)*delta_n*Variable(n_d_time)) * Variable(nh_time_mask)).sum(dim = 2)

       
        return p_lambda, n_lambda

    def loss_func(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask, h_ab, h_cb, h_p, nh_nodes, nh_times, nh_time_mask, nh_ab, nh_cb, nh_p):
        if torch.cuda.is_available() and 0:
            with torch.cuda.device(DID):
                p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask, h_ab, h_cb, h_p, nh_nodes, nh_times, nh_time_mask, nh_ab, nh_cb, nh_p)
                loss = -torch.log(p_lambdas.sigmoid() + 1e-6) - torch.log(
                    n_lambdas.neg().sigmoid() + 1e-6).sum(dim=1)

        else:
            p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask, h_ab, h_cb, h_p, nh_nodes, nh_times, nh_time_mask, nh_ab, nh_cb, nh_p)
            loss = -torch.log(p_lambdas.sigmoid() + 1e-6) - torch.log(
                    n_lambdas.neg().sigmoid() + 1e-6).sum(dim=1)
        return loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask, h_ab, h_cb, h_p, nh_nodes, nh_times, nh_time_mask, nh_ab, nh_cb, nh_p):
        if torch.cuda.is_available() and 0:
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask, h_ab, h_cb, h_p, nh_nodes, nh_times, nh_time_mask, nh_ab, nh_cb, nh_p)
                loss = loss.sum()
                self.loss += loss.data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask, h_ab, h_cb, h_p, nh_nodes, nh_times, nh_time_mask, nh_ab, nh_cb, nh_p)
            loss = loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch,
                                shuffle=True, num_workers=5)
            if epoch % self.save_step == 0 and epoch != 0:
                #torch.save(self, './model/dnrl-dblp-%d.bin' % epoch)
                self.save_node_embeddings('./emb/school/att_school_motif_histlen27_%d_64.emb' % (epoch))

            for i_batch, sample_batched in enumerate(loader):
                if i_batch % 100 == 0 and i_batch != 0:
                    print('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)) + '\tdelta:' + str(
                        self.delta.mean().cpu().data.numpy()))
                    #sys.stdout.flush()

                if torch.cuda.is_available() and 0:
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['target_time'].type(FType).cuda(),
                                    sample_batched['neg_nodes'].type(LType).cuda(),
                                    sample_batched['history_nodes'].type(LType).cuda(),
                                    sample_batched['history_times'].type(FType).cuda(),
                                    sample_batched['history_masks'].type(FType).cuda(),
                                    sample_batched['history_ab'].type(LType).cuda(),
                                    sample_batched['history_cb'].type(LType).cuda(),
                                    sample_batched['history_p'].type(LType).cuda(),                                    									
                                    sample_batched['neg_h_nodes'].type(LType).cuda(),
                                    sample_batched['neg_h_times'].type(FType).cuda(),
                                    sample_batched['neg_h_masks'].type(FType).cuda(),
                                    sample_batched['neg_h_ab'].type(LType).cuda(),
                                    sample_batched['neg_h_cb'].type(LType).cuda(),
                                    sample_batched['neg_h_p'].type(LType).cuda()) 									
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType),
                                sample_batched['history_ab'].type(LType),
                                sample_batched['history_cb'].type(LType),
                                sample_batched['history_p'].type(LType),								
                                sample_batched['neg_h_nodes'].type(LType),
                                sample_batched['neg_h_times'].type(FType),
                                sample_batched['neg_h_masks'].type(FType),
                                sample_batched['neg_h_ab'].type(LType),
                                sample_batched['neg_h_cb'].type(LType),
                                sample_batched['neg_h_p'].type(LType))								

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data)) + '\n')
            sys.stdout.flush()

        self.save_node_embeddings('./emb/school/att_school_motif_histlen27_%d_64.emb' % (self.epochs))

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()


if __name__ == '__main__':
    htne = MTNE_a('./data/school/rm_school.txt', directed=False, epoch_num = 20)
    htne.train()
