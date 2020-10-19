"""
Prepares the dictionary with image mu, std and text mu by combining the
separately outputs
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/raw'


def prepare_data():
    with open(f'{DATA_PATH}/image_mu.pkl', 'rb') as f:
        image_mu = pickle.load(f)  # dict
    with open(f'{DATA_PATH}/image_std.pkl', 'rb') as f:
        image_std = pickle.load(f)  # dict
    with open(f'{DATA_PATH}/sentence_latent.pkl', 'rb') as f:
        text_mu = pickle.load(f)  # list of dict

    modified_img_mu = {k.replace('_', '').replace('-', '').replace('.png', ''): v
                     for k, v in image_mu.items()}
    modified_img_std = {k.replace('_', '').replace('-', '').replace('.png', ''): v
                        for k, v in image_std.items()}

    data = []
    not_found = 0
    for o in text_mu:
        spec_id = o['audioClip']
        # a number of the form 000/009/010/023 has an extra 0 (except that album)
        if 'NineInchNails1000' not in spec_id and spec_id[-3] == '0':
            spec_id = spec_id.replace(spec_id[-3:], f'{int(spec_id[-2:])}')

        try:
            data.append({'spec_mu': modified_img_mu[spec_id],
                        'spec_std': modified_img_std[spec_id],
                        'lyrics_mu': np.array([o['z']]),
                        'lyrics': o['line']})
        except KeyError as e:
            not_found += 1

    print(len(data))

    train, val = train_test_split(data, test_size=0.2)
    print(f'Total {not_found} missing specs')
    with open(f'data/processed/gan_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open(f'data/processed/train_data.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(f'data/processed/val_data.pkl', 'wb') as f:
        pickle.dump(val, f)

if __name__ == '__main__':
    prepare_data()
