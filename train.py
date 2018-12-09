# Python packages
import time
import argparse
import pickle
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext import data
# My packages
import modelbuilder
import trainer

# Parse command line options.
def arg_parse():
    parser = argparse.ArgumentParser(description='train.py')
    
    # Data load and save
    parser.add_argument('-src', required=True, help='Path to the source data pickle file')
    parser.add_argument('-tgt', required=True, help='Path to the target data pickle file')
    parser.add_argument('-glove', required=True, help='Path to GLoVE embedding pickle file')
    parser.add_argument('-save', required=True, help='Save path for output')
    
    # Data processing
    parser.add_argument('-src_vocab_size', type=int, default=45000, help="Size of source vocabulary")
    parser.add_argument('-tgt_vocab_size', type=int, default=28000, help="Size of target vocabulary")
    parser.add_argument('-src_seq_len', type=int, default=90, help="Max source sequence length")
    parser.add_argument('-tgt_seq_len', type=int, default=25, help="Max source sequence length")
    parser.add_argument('-trn_fract', type=float, default=0.9, help="Fraction of data split for training")
    parser.add_argument('-batch_size', type=int, default=64, help="Batch size")

    # Model parameters
    parser.add_argument('-bidir', type=bool, default=False, help="First layer of encoder is bidrectional")
    parser.add_argument('-num_hid', type=int, default=600, help="Number of hidden units in the encoder and decoder")
    parser.add_argument('-num_layers', type=int, default=2, help="Needs to be at least 2 if including dropout in LSTM")
    parser.add_argument('-cell_drop', type=float, default=0.3, help="Dropout between memory cell (LSTM) stacks")
    parser.add_argument('-emb_drop', type=float, default=0, help="Embedding layer dropout")
    parser.add_argument('-fc_drop', type=float, default=0.3, help="Finally fully connected layer dropout")
    parser.add_argument('-grad_clip', type=float, default=5.0, help="Gradient clipping value")

    # Training parameters
    parser.add_argument('-lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('-train_steps', type=int, default=15000, help="Number of training iterations (minibatches)")
    parser.add_argument('-val_freq', type=int, default=1100, help="Run validation every X iterations")
    parser.add_argument('-print_freq', type=int, default=500, help="Report training stats every X iterations")
    
    return parser.parse_args()

# Load and trim the data. Limits sequence length (RNN steps) based on command line arguments
def load_trim(opt):
    # Load tokenized data
    src_tok = pickle.load(open(opt.src,'rb'))
    tgt_tok = pickle.load(open(opt.tgt,'rb'))

    # Keep sentences where source is within limit
    keep_src = np.array([len(s) < opt.src_seq_len for s in src_tok])
    src_tok_trim = np.array(src_tok)[keep_src]
    tgt_tok_trim = np.array(tgt_tok)[keep_src]

    # Same with target sequence length
    keep_tgt = np.array([len(s) < opt.tgt_seq_len for s in tgt_tok_trim])
    src_tok_trim = np.array(src_tok_trim)[keep_tgt]
    tgt_tok_trim = np.array(tgt_tok_trim)[keep_tgt]

    return src_tok_trim, tgt_tok_trim


# Build up the iterator to use with model, the vocab, and the fields
def build_iter(opt, src_tok, tgt_tok, device):
    src_field = data.Field(init_token='<sos>', eos_token='<eos>', lower=True)
    tgt_field = data.Field(init_token='<sos>', eos_token='<eos>', lower=True)
    fields = [("src", src_field), ("tgt", tgt_field)]

    examples = [data.Example.fromlist([src_tok[i], tgt_tok[i]], fields) for i in range(len(src_tok))]
    dataset = data.Dataset(examples, fields)
    trn_ds, val_ds = dataset.split(split_ratio=opt.trn_fract)

    src_field.build_vocab(trn_ds, max_size=opt.src_vocab_size)
    tgt_field.build_vocab(trn_ds, max_size=opt.tgt_vocab_size)

    trn_dl, val_dl = data.BucketIterator.splits((trn_ds, val_ds), sort=True, sort_key=lambda x: len(x.src), \
                                       batch_size=opt.batch_size, device=device)

    return src_field, tgt_field, trn_dl, val_dl


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = arg_parse()

    # Load data and remove sequences longer than limits
    src_tok_trim, tgt_tok_trim = load_trim(opt)
    print(f'Trimmed tokenize check. Length src: {len(src_tok_trim)}, length tgt: {len(tgt_tok_trim)}')

    # GLoVE pretrained word embeddings
    glove = pickle.load(open(opt.glove,'rb'))
    opt.emb_dim = len(glove['the'])

    # Create source field objects (contain vocab) and iterators
    src_field, tgt_field, trn_dl, val_dl = build_iter(opt, src_tok_trim, tgt_tok_trim, device)
    print(f'Src vocab length: {len(src_field.vocab)}, Tgt vocab length: {len(tgt_field.vocab)}')
    print(f'Train DL length: {len(trn_dl)}, Val DL length: {len(val_dl)}')

    # Build the model
    encoder = modelbuilder.Encoder(glove, src_field.vocab.itos, opt)
    decoder = modelbuilder.Decoder(glove, tgt_field.vocab.itos, opt)
    model = modelbuilder.Seq2Seq(encoder, decoder, opt.num_hid, len(tgt_field.vocab.itos), opt.cell_drop).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    compute = trainer.LossCompute(model.generator, tgt_field.vocab)

    # Train
    trainer = trainer.Trainer(model, compute, optimizer, opt)
    trainer.train(trn_dl, val_dl, opt)
