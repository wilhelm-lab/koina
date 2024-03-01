import argparse
from pathlib import Path
import numpy as np
from .grpc import Koina
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument("-u", "--url", default="koina.wilhelmlab.org")
parser.add_argument("-m", "--model")
parser.add_argument("--no-ssl", action="store_false", dest="ssl")
parser.add_argument(
    "-i", "--input", help="Pattern to read input arrays", default="*.npy"
)


args = parser.parse_args()

print(args)
size = 20000

print(glob(args.input))

# client = Koina(server_url=args.url, model_name=args.model, ssl=args.ssl)
# input_data = {
#     "peptide_sequences": np.load(),
#     "precursor_charges": np.array([2 for _ in range(size)]),
#     "collision_energies": np.array([20 for _ in range(size)]),
#     "fragmentation_types": np.array(["HCD" for _ in range(size)]),
#     "instrument_types": np.array(["QE" for _ in range(size)]),
# }
# predictions = client.predict(input_data)
