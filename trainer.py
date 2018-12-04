import math
import torch
import torch.nn as nn

# Loss compute object
class LossCompute():
    def __init__(self, generator, tgt_vocab):
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.pad_idx = self.tgt_vocab.stoi['<pad>']
        self.criterion = nn.NLLLoss(ignore_index=self.pad_idx, reduction='sum')
        
    def reshape(self, outp):
        return outp.view(-1, outp.size(2))
    
    def compute_loss(self, output, target):
        # Output here is 
        reshaped_output = self.reshape(output)
        scores = self.generator(reshaped_output)   # Reshapes from bs*sl x hid_dim to bs*sl x vocab_len
        gtruth = target[1:].view(-1)     # This is different than what they do. Look into best way to do this.
        loss = self.criterion(scores, gtruth)
        return loss
    
    def train_compute_loss(self, output, target, batch_size):
        loss = self.compute_loss(output, target)
        loss.div(float(batch_size)).backward()    # Backward pass
        
        not_pad = target.ne(self.pad_idx)
        not_pad = not_pad.sum().item()
        
        return loss.item(), not_pad

class Trainer():
    def __init__(self, model, loss_compute, optim, gradclip):
        self.model = model
        self.loss_compute = loss_compute
        self.optim = optim
        self.model.train()
        self.gradclip = gradclip
        
    def train(self, train_dl, valid_dl, opt):
        step = 1
        while step <= opt.train_steps:
            train_loss = 0
            train_words = 0
            for i, batch in enumerate(train_dl):
                # do the forward pass
                batch_loss, batch_words = self.train_func(batch)
                train_loss += batch_loss
                train_words += batch_words
                # Report training progress
                if step % opt.print_freq == 0:
                    train_loss /= train_words
                    print(f'Iteration: {step}, Training loss: {train_loss:.3f}, Training PPL: {math.exp(train_loss):.3f}')
                    train_loss = 0
                    train_words = 0
                    
                # Run validation
                if step % opt.val_freq == 0:
                    self.val_func(valid_dl)
                
                step += 1
                if step > opt.train_steps:
                    break            
    
    def train_func(self, batch):
        self.model.zero_grad()
        output = self.model(batch.src, batch.tgt) # Run forward pass up to generator
        loss, words = self.loss_compute.train_compute_loss(output, batch.tgt, batch.batch_size) # Run through generator and compute loss

        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradclip) # Gradient clip
        self.optim.step() # Model update step
        return loss, words
    
    def val_func(self, valid_dl):
        self.model.eval()
        val_loss = 0
        val_words = 0
        print("Running validation ...")
        with torch.no_grad():
            for i, batch in enumerate(valid_dl):
                sl,bs = batch.tgt.shape
                output = self.model(batch.src, batch.tgt, 0) # Run forward pass up to generator
                loss = self.loss_compute.compute_loss(output, batch.tgt)
                val_loss += loss.item()
                
                not_pad = batch.tgt.ne(self.loss_compute.pad_idx)
                not_pad = not_pad.sum().item()
                val_words += not_pad
            val_loss /= val_words
            print(f'Validation Loss: {val_loss:.3f}, Validation PPL: {math.exp(val_loss):.3f}')
        self.model.train()