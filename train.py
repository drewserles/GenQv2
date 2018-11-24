import argparse
import pickle
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext import data

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
    keep_tgt = np.array([len(s) < tgt_seq_len for s in tgt_tok_trim])
    src_tok_trim = np.array(src_tok_trim)[keep_tgt]
    tgt_tok_trim = np.array(tgt_tok_trim)[keep_tgt]

    return src_tok_trim, tgt_tok_trim

# Create a list 'itos' where each items is a word in the vocab and its index is the numerical representation
# Also create a dictionary stoi that converts from a string to its integer value. key:value
# Finally, convert the tokenized data to be integers rather than strings so they can be fed into the model.
# def gen_ids(tok, vocab_size):
#     freq = collections.Counter(p for o in tok for p in o)
#     itos = [o for o,c in freq.most_common(vocab_size)]
#     itos.insert(0, '_ph_')
#     itos.insert(1, '_pad_')
#     itos.insert(2, '_bos_')
#     itos.insert(3, '_eos_')
#     itos.insert(4, '_unk_')
#     stoi = collections.defaultdict(lambda: 4, {v:k for k,v in enumerate(itos)})
#     ids = np.array([([2] + [stoi[o] for o in p] + [3]) for p in tok])

#     return ids, itos, stoi

# # Randomly split up the input and target data into training and validation sets.
# # Return 4 numpy arrays - training input, training targets, val input, and val targets
# def train_val_split(inp_data, out_data, train_fraction):
#     trn_idx = np.random.rand(len(inp_data)) < train_fraction
    
#     inp_trn = inp_data[trn_idx]
#     inp_test = inp_data[~trn_idx]
    
#     outp_val = out_data[trn_idx]
#     outp_val = out_data[~trn_idx]
#     return inp_trn, outp_trn, inp_val, outp_val
    
# # Build dataset - extension of PyTorch dataset class
# class Seq2SeqDataset(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#     def __getitem__(self, idx):
#         return [np.array(self.x[idx]), np.array(self.y[idx])]
#     def __len__(self):
#         return len(self.x)

# Build up the iterator to use with model, the vocab, and the fields
def build_iter(opt, src_tok, tgt_tok, device):
    src_field = data.Field(init_token='<sos>', eos_token='<eos>', lower=True)
    tgt_field = data.Field(init_token='<sos>', eos_token='<eos>', lower=True)
    fields = [("src", src_field), ("tgt", tgt_field)]

    examples = [data.Example.fromlist([src_tok[i], tgt_tok[i]], fields) for i in range(len(src_tok))]
    dataset = data.Dataset(examples, fields)
    trn_ds, val_ds = mydataset.split(split_ratio=opt.trn_fract)

    src_field.build_vocab(trn_ds, max_size=opt.src_vocab_size)
    tgt_field.build_vocab(trn_ds, max_size=opt.tgt_vocab_size)

    trn_dl, val_dl = BucketIterator.splits((trn_ds, val_ds), sort_key=lambda x: len(x.src), \
                                       batch_size=opt.batch_size, device=device)

    return src_field, tgt_field, trn_dl, val_dl

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    options = arg_parse()
    src_tok_trim, tgt_tok_trim = load_trim(options)
    print(f'Trimmed tokenize check. Length src: {len(src_tok_trim)}, length tgt: {len(tgt_tok_trim)}')

    src_field, tgt_field, trn_dl, val_dl = build_iter(options, src_tok_trim, tgt_tok_trim, device)

    # This is the previous method. Using torchtext now
    # src_ids, src_itos, src_stoi = gen_ids(src_tok_trim, options.src_vocab_size)
    # tgt_ids, tgt_itos, tgt_stoi = gen_ids(tgt_tok_trim, options.tgt_vocab_size)
    # print(f'Trimmed ids check. Length src: {len(src_ids)}, length tgt: {len(tgt_ids)}')

    # src_trn_ids, tgt_trn_ids, src_val_ids, tgt_val_ids = train_val_split(src_ids, tgt_ids, options.trn_fract)

    # # Build dataset
    # trn_ds = Seq2SeqDataset(src_trn_ids, tgt_trn_ids)
    # val_ds = Seq2SeqDataset(src_val_ids, tgt_val_ids)
    # # Build data loader. Work to be done here.


