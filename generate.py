import argparse

# Parse command line options.
def arg_parse():
    parser = argparse.ArgumentParser(description='generate.py')

    # Data load and save
    parser.add_argument('-model', required=True, help='Path to the pretrained model file')
    parser.add_argument('-field', required=True, help='Path to the saved field file')
    parser.add_argument('-src', required=True, help='Path to the source data to generate from')
    parser.add_argument('-output', required=True, help='Output save location')

    # Generation parameters
    parser.add_argument('-batch_size', type=int, default=64, help="Batch size")
    
def load_data(opt):
    # Load the (already tokenized) data
    with open(opt.src, 'r') as f:
        src_tok = [line.rstrip('\n') for line in f]
    
    # Load model
    model = torch.load(opt.model)
    fields = torch.load(opt.field)
    src_field = fields['src_field']
    tgt_field = fields['tgt_field']
    
    # Create dataloader
    fields = [("src", src_field), ("tgt", tgt_field)]
    # Need to figure out what to do if there's no target dataset. This is a hack for now recycline the src
    examples = [data.Example.fromlist([src_tok[i], src_tok[i]], fields) for i in range(len(src_tok))]
    
    ds = data.Dataset(examples, fields)
    dl = data.Iterator(dataset=ds, device=opt.device, batch_size=opt.batch_size, train=False, sort=False,\
                       sort_within_batch=False, shuffle=False)
    return model, src_field, tgt_field, dl

if __name__ == "__main__":
    opt = arg_parse()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    model, src_field, tgt_field, gen_dl = load_data(opt)
    
    