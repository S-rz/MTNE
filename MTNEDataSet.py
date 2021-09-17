from torch.utils.data import Dataset
import numpy as np
import sys
import random


class MTNEDataSet(Dataset):
    def __init__(self, file_path, neg_size, hist_len, directed=False, transform=None):
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.directed = directed
        self.transform = transform

        self.max_d_time = -sys.maxsize  # Time interval [0, T]

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)

        self.node2hist = dict()
        self.node_set = set()
        self.degrees = dict()
        with open(file_path, 'r') as infile:
            for line in infile:
                parts = line.split()
                s_node = int(parts[0])  # source node
                t_node = int(parts[1])  # target node
                d_time = float(parts[2])  # time slot, delta t

                self.node_set.update([s_node, t_node])

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, d_time, 0))

                #if not directed:
                if t_node not in self.node2hist:
                    self.node2hist[t_node] = list()
                self.node2hist[t_node].append((s_node, d_time, 1))

                if d_time > self.max_d_time:
                    self.max_d_time = d_time

                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0
                self.degrees[s_node] += 1
                self.degrees[t_node] += 1

        self.node_dim = len(self.node_set)

        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()

    def get_node_dim(self):
        return self.node_dim

    def get_max_d_time(self):
        return self.max_d_time

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        for k in range(self.node_dim):
            tot_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]

        hist = self.get_triad_node(s_node, t_node, t_idx, t_time)
        
        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]

        hist_ab = [h[2] for h in hist]
        hist_cb = [h[3] for h in hist]
        hist_p = [h[4] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        np_h_ab = np.zeros((self.hist_len,))
        np_h_ab[:len(hist_ab)] = hist_ab

        np_h_cb = np.zeros((self.hist_len,))
        np_h_cb[:len(hist_cb)] = hist_cb

        np_h_p = np.zeros((self.hist_len,))
        np_h_p[:len(hist_p)] = hist_p

        neg_nodes = self.negative_sampling()
        n_h_nodes = list()
        n_h_times = list()
        n_h_masks = list()
        n_h_ab = list()
        n_h_cb = list()
        n_h_p = list()
        for n in neg_nodes:
            n_hist = self.get_triad_node(s_node, n, t_idx, t_time)
            n_hist_nodes = [h[0] for h in n_hist]
            n_hist_times = [h[1] for h in n_hist]

            n_hist_ab = [h[2] for h in n_hist]
            n_hist_cb = [h[3] for h in n_hist]
            n_hist_p = [h[4] for h in n_hist]

            n_np_h_nodes = np.zeros((self.hist_len,))
            n_np_h_nodes[:len(n_hist_nodes)] = n_hist_nodes
            n_np_h_times = np.zeros((self.hist_len,))
            n_np_h_times[:len(n_hist_times)] = n_hist_times
            n_np_h_masks = np.zeros((self.hist_len,))
            n_np_h_masks[:len(n_hist_nodes)] = 1.

            n_np_h_ab = np.zeros((self.hist_len,))
            n_np_h_ab[:len(n_hist_ab)] = n_hist_ab
            n_np_h_cb = np.zeros((self.hist_len,))
            n_np_h_cb[:len(n_hist_cb)] = n_hist_cb
            n_np_h_p = np.zeros((self.hist_len,))
            n_np_h_p[:len(n_hist_p)] = n_hist_p

            n_h_nodes.append(n_np_h_nodes.tolist())
            n_h_times.append(n_np_h_times.tolist())
            n_h_masks.append(n_np_h_masks.tolist())
            n_h_ab.append(n_np_h_ab.tolist())
            n_h_cb.append(n_np_h_cb.tolist())
            n_h_p.append(n_np_h_p.tolist())

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_masks': np_h_masks,
            'history_ab': np_h_ab,
            'history_cb': np_h_cb,
            'history_p': np_h_p,
            'neg_nodes': neg_nodes,
            'neg_h_nodes': np.array(n_h_nodes),
            'neg_h_times': np.array(n_h_times),
            'neg_h_masks': np.array(n_h_masks),
            'neg_h_ab': np.array(n_h_ab),   			
            'neg_h_cb': np.array(n_h_cb),
            'neg_h_p': np.array(n_h_p),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    def get_triad_node(self, s_node, t_node, t_idx, t_time):
        hist_s = self.node2hist[s_node][0:t_idx]
        hist_t = self.node2hist[t_node]
        hist = list()
        d = {}	
        for s in hist_s:
            d[s[0]] = [s[1], s[2], -1, -1]
        for t in hist_t:
            if t[1] > t_time:
                break
            elif t[0] in d and t[1] < t_time:
                d[t[0]][2] = t[1]
                d[t[0]][3] = t[2]	
        for k in d:
            if d[k][-1] != -1:		
                if d[k][0] < d[k][2]:
                    pattern = 1
                    time = d[k][2]
                else:
                    pattern = 0
                    time = d[k][0]
                hist.append((k, time, d[k][1], d[k][3], pattern))						
        l = len(hist)
        if l > self.hist_len:
            hist = hist[l - self.hist_len : l]
        return hist			
    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
        sampled_nodes = self.neg_table[rand_idx]
        return sampled_nodes
