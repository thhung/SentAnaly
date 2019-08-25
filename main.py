#####################################################################
## Result for normal settings: 87 % 


import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from ranger import Ranger

# import model as pkgmodel


def main():

    SEED = 1234

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    TEXT = data.Field(lower=True, batch_first=True, tokenize = 'spacy')
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='/tmp/imdb/')
    train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=100, cache='/tmp/glove/'), 
                    unk_init = torch.Tensor.normal_
    )

    LABEL.build_vocab(train_data)

    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE, 
        device = device)
    vocab_size, embedding_dim = TEXT.vocab.vectors.shape

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
            self.conv = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=embedding_dim,
                                                out_channels=num_filters,
                                                kernel_size=k, stride=1),
                                                nn.Dropout(p=0.5, inplace=True)) for k in kernel_sizes])
            # self.dropout = nn.Dropout(d_prob)
            self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

        def forward(self, x):
            batch_size, sequence_length = x.shape
            x = self.embedding(x).transpose(1, 2)
            x = [F.relu(conv(x)) for conv in self.conv]
            x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
            x = torch.cat(x, dim=1)
            # x = self.fc(self.dropout(x))
            x = self.fc(x)
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

    model = SentimentAnalysisCNN(vocab_size=vocab_size, #pkgmodel
                    embedding_dim=embedding_dim,
                    kernel_sizes=[3, 4, 5],
                    num_filters=100,
                    num_classes=1, 
                    d_prob=0.5,
                    mode='static')
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    optimizer = Ranger(model.parameters(), weight_decay=0.1)
    criterion = nn.BCELoss()

    def process_function(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch.text, batch.label
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_function(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch.text, batch.label
            y_pred = model(x)
            return y_pred, y

    trainer = Engine(process_function)
    train_evaluator = Engine(eval_function)
    validation_evaluator = Engine(eval_function)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    def thresholded_output_transform(output):
        y_pred, y = output
        y_pred = torch.round(y_pred)
        return y_pred, y
    Accuracy(output_transform=thresholded_output_transform).attach(train_evaluator, 'accuracy')
    Loss(criterion).attach(train_evaluator, 'bce')

    Accuracy(output_transform=thresholded_output_transform).attach(validation_evaluator, 'accuracy')
    Loss(criterion).attach(validation_evaluator, 'bce')

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer, ['loss'])

    def score_function(engine):
        val_loss = engine.state.metrics['bce']
        return -val_loss

    handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    validation_evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_iterator)
        metrics = train_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_bce = metrics['bce']
        pbar.log_message(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_bce))
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        validation_evaluator.run(valid_iterator)
        metrics = validation_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_bce = metrics['bce']
        pbar.log_message(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_bce))
        pbar.n = pbar.last_print_n = 0

    checkpointer = ModelCheckpoint('/tmp/models', 'textcnn_ranger_wd_0_1', save_interval=1, n_saved=2, create_dir=True, save_as_state_dict=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'textcnn_ranger_wd_0_1': model})
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
    
    trainer.run(train_iterator, max_epochs=20)



if __name__ == "__main__":
    main()

