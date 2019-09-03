## This file is kept for reference. Please refer to the main.py. 

import torch.nn as nn
import torch.nn.functional as F


class SentimentAnalysisCNN(nn.Module):
    """ Network architecture
    """
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters, num_classes, d_prob, mode):
        """ Args:
                vocab_size : int - size of vocabulary in dictionary
                embedding_dim : int - the dimension of word embedding vector
                kernel_sizes : list of int - sequence of sizes of kernels in this architecture
                num_filters : 
                num_classes : int - number of classes to classify
                d_prob: probability for dropout layer
                mode:  one of :
                        static      : pretrained weights, non-trainable
                        nonstatic   : pretrained weights, trainable
                        rand        : random init weights
        """
        super(SentimentAnalysisCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.d_prob = d_prob
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.load_embeddings()
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                             out_channels=num_filters,
                                             kernel_size=k, stride=1) for k in kernel_sizes])
        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = self.embedding(x).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.fc(self.dropout(x))
        return torch.sigmoid(x).squeeze()

    def load_embeddings(self):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(TEXT.vocab.vectors)
            if 'non' not in self.mode:
                self.embedding.weight.data.requires_grad = False
                print('Loaded pretrained embeddings, weights are not trainable.')
            else:
                self.embedding.weight.data.requires_grad = True
                print('Loaded pretrained embeddings, weights are trainable.')
        elif self.mode == 'rand':
            print('Randomly initialized embeddings are used.')
        else:
            raise ValueError('Unexpected value of mode. Please choose from static, nonstatic, rand.')
