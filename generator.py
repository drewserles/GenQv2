import torch

class Generator():
    def __init__(self, model, src_field, tgt_field):
        self.model = model
        self.src_field = src_field
        self.tgt_field = tgt_field
        # Set to eval
        self.model.eval()
        
    def generate(self, gen_dl):
        
        with torch.no_grad():
            for i, batch in enumerate(gen_dl):
                # Outputs up to generator
                output = self.model(batch.src, batch.tgt, 0)
                # Run through the generator
                output = self.model.generator(output)
                # Take the max token
                print(output.shape)
                output = output.max(1)[1]
                print(output)
                return output