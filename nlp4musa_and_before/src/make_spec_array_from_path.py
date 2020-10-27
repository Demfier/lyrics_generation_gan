import os
import pickle
from tqdm import tqdm
from utils.preprocess import read_spectrogram


def make_spec_array_pkl(spec_root_dir, save_path):
    spec_files = os.listdir(spec_root_dir)
    spec_array = {}
    for f in tqdm(spec_files):
        # skip a file which isn't an image
        if not f.endswith('.png'):
            continue
        spec_array[str(f)] = read_spectrogram(spec_root_dir + f)
    with open(save_path, 'wb') as f:
        pickle.dump(spec_array, f)


if __name__ == '__main__':
    SPEC_PATH = '/collection/d35kumar/lyr_gen_neurips/data/Instrumental/DepecheMode/Specs/'
    SAVE_PATH = 'data/processed/neurips/instrumental_dm.pkl'
    make_spec_array_pkl(SPEC_PATH, SAVE_PATH)
