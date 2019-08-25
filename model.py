import torch.nn as nn
import torch.nn.functional as F


class SentimentAnalysisCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters, num_classes, d_prob, mode):
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

# class SentimentAnalysisCNN(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
#                  dropout, pad_idx):
        
#         super().__init__()
        
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
#         self.conv_0 = nn.Conv2d(in_channels = 1, 
#                                 out_channels = n_filters, 
#                                 kernel_size = (filter_sizes[0], embedding_dim))
        
#         self.conv_1 = nn.Conv2d(in_channels = 1, 
#                                 out_channels = n_filters, 
#                                 kernel_size = (filter_sizes[1], embedding_dim))
        
#         self.conv_2 = nn.Conv2d(in_channels = 1, 
#                                 out_channels = n_filters, 
#                                 kernel_size = (filter_sizes[2], embedding_dim))
        
#         self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, text):
        
#         #text = [sent len, batch size]
        
#         text = text.permute(1, 0)
                
#         #text = [batch size, sent len]
        
#         embedded = self.embedding(text)
                
#         #embedded = [batch size, sent len, emb dim]
        
#         embedded = embedded.unsqueeze(1)
        
#         #embedded = [batch size, 1, sent len, emb dim]
        
#         conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
#         conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
#         conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
            
#         #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
#         pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
#         pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
#         pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
#         #pooled_n = [batch size, n_filters]
                
#         cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))

#         #cat = [batch size, n_filters * len(filter_sizes)]
            
#         return self.fc(cat)


# pretrained_embeddings = TEXT.vocab.vectors

# model.embedding.weight.data.copy_(pretrained_embeddings)

# UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

# model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
# model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)