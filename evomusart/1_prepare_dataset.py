"""
This script re-prepares data for EvoMusArt submission. Spectrograms embeddings
in the GAN data need to be repalced with their
"""
import pickle
import torch
import vae
import numpy as np
import scipy.misc, skimage

OLD_GAN_TRAIN_DATA = '../data/processed/train_data_mar29_500epoch.pkl'
OLD_GAN_VAL_DATA = '../data/processed/valid_data_mar29_500epoch.pkl'
FIXED_SPEC_ARRAY = '../data/processed/spec_array_fix_stretching_training.pkl'

NEW_GAN_TRAIN_DATA = '../data/processed/train_data_nov14-21_500epoch.pkl'
NEW_GAN_VAL_DATA = '../data/processed/valid_data_nov14-21_500epoch.pkl'

SPEC_VAE_PATH = '/collection/ovechtom/ghosts-two-2VAE/ICCC/vae-stretching_fixed-1e4_200-200.torch'
print('Loading spec vae in eval mode...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SPEC_VAE = vae.VAE().to(device)
SPEC_VAE.load_state_dict(torch.load(SPEC_VAE_PATH, map_location=device))
SPEC_VAE.eval()


def read_pickle(path):
    print(f'Loading {path}...')
    return pickle.load(open(path, 'rb'))


def write_pickle(f, path):
    pickle.dump(f, open(path, 'wb'))
    print(f'New data saved at {path}...')


def process_spectrogram(img):
    img =  np.array(img)
    # print(img.shape)
    img = img/255.0
    img = img.astype('float32')
    img = img.transpose(2,0,1)
    return img


# takes in the spectrogram, runs the VAE on inference mode and returns mu and std
def sample_z(img):
    with torch.no_grad():
        x = torch.from_numpy(img)
        x = x.to(device)
        x = x.unsqueeze(0)
        z, mu, logvar = SPEC_VAE.encode(x)
        std = logvar.mul(0.5).exp_()
        std = std.cpu().numpy()
        mu = mu.cpu().numpy()
    return mu, std


def fix_streching(data, spec_array, olga_to_me_key_map):
    print(f'#old data: {len(data)}')
    print(f'#spec arrays: {len(spec_array)}')
    for idx, o in enumerate(data):
        # old data has olga's nomenclature
        spec_id = o['spec_id']
        # get my nomenclature
        my_spec_id = olga_to_me_key_map[spec_id]
        img = process_spectrogram(spec_array[my_spec_id])
        # get mu and std for the new spec array
        mu, std = sample_z(img)
        o['spec_mu'], o['spec_std'], o['my_spec_id'] = mu, std, my_spec_id
        data[idx] = o
    return data


def main():
    train, val = read_pickle(OLD_GAN_TRAIN_DATA), read_pickle(OLD_GAN_VAL_DATA)
    spec_array = read_pickle(FIXED_SPEC_ARRAY)

    # transform spec ids to olga's nomenclature
    olga_to_me_key_map = {}
    temp_spec_array = {}
    for k, v in spec_array.items():
        olga_spec_id = k.replace('_', '').replace('-', '').replace('.png', '')
        temp_spec_array[olga_spec_id] = v
        olga_to_me_key_map[olga_spec_id] = k

    write_pickle(fix_streching(train, spec_array, olga_to_me_key_map), NEW_GAN_TRAIN_DATA)
    write_pickle(fix_streching(val, spec_array, olga_to_me_key_map), NEW_GAN_VAL_DATA)



if __name__ == '__main__':
    main()
