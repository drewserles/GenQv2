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

    # Training parameters
    parser.add_argument('-lr', type=float, default=0.001, help="Learning rate")
    
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
    EMB_DIM = len(glove['the'])

    # Create source field objects (contain vocab) and iterators
    src_field, tgt_field, trn_dl, val_dl = build_iter(opt, src_tok_trim, tgt_tok_trim, device)
    print(f'Src vocab length: {len(src_field.vocab)}, Tgt vocab length: {len(tgt_field.vocab)}')
    print(f'Train DL length: {len(trn_dl)}, Val DL length: {len(val_dl)}')

    # Build the model
    encoder = modelbuilder.Encoder(glove, src_field.vocab.itos, EMB_DIM, opt.num_hid, \
                                opt.num_layers, opt.cell_drop, opt.bidir)
    decoder = modelbuilder.Decoder(glove, tgt_field.vocab.itos, EMB_DIM, opt.num_hid, \
                                opt.num_layers, opt.cell_drop)
    model = modelbuilder.Seq2Seq(encoder, decoder, opt.fc_drop)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    compute = trainer.LossCompute(model.generator, tgt_field.vocab)

    # Run in training
    CLIP = 5
    TRAIN_STEPS = 15000         # len(trn_dl)=1120, so 1120 steps(batches) per epoch
    VAL_FREQ = 1100
    PRINT_FREQ = 500
    best_val_loss = float('inf')

    trainer = trainer.Trainer(model, compute, optimizer, CLIP)
    trainer.train(trn_dl, val_dl, TRAIN_STEPS, PRINT_FREQ, VAL_STEPS)

    # for epoch in range(N_EPOCHS):
    #     start = time.time()
    #     train_loss = modelbuilder.train(model, trn_dl, optimizer, crit, CLIP, REPORT_FREQ)
    #     valid_loss = modelbuilder.evaluate(model, val_dl, crit)
    #     stop = time.time()
        
    #     if valid_loss < best_val_loss:
    #         best_val_loss = valid_loss
    #         torch.save(model.state_dict(), opt.save + 'Best_Val_Model.pt')
        
    #     print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} \
    #         | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} | Epoch time: {stop - start:.1f} sec')






    # This is the previous method. Using torchtext now
    # src_ids, src_itos, src_stoi = gen_ids(src_tok_trim, options.src_vocab_size)
    # tgt_ids, tgt_itos, tgt_stoi = gen_ids(tgt_tok_trim, options.tgt_vocab_size)
    # print(f'Trimmed ids check. Length src: {len(src_ids)}, length tgt: {len(tgt_ids)}')

    # src_trn_ids, tgt_trn_ids, src_val_ids, tgt_val_ids = train_val_split(src_ids, tgt_ids, options.trn_fract)

    # # Build dataset
    # trn_ds = Seq2SeqDataset(src_trn_ids, tgt_trn_ids)
    # val_ds = Seq2SeqDataset(src_val_ids, tgt_val_ids)
    # # Build data loader. Work to be done here.


