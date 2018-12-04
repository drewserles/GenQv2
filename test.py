import argparse

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-bridge', action="store_true",
                       help="""Have an additional layer between the last encoder
                       state and the first decoder state""")

options = parser.parse_args()
print(options.bridge)
options.emb_dim = 90
print(options.emb_dim)