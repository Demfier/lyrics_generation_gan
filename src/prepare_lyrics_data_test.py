"""
Prepares the dictionary with image mu, std and text mu by combining the
separately outputs
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/raw'


def prepare_data():
    with open(f'{DATA_PATH}/image_mu_testing_ISMIR.pkl', 'rb') as f:
        image_mu = pickle.load(f)  # dict
    with open(f'{DATA_PATH}/image_std_testing_ISMIR.pkl', 'rb') as f:
        image_std = pickle.load(f)  # dict

    data = []
    not_found = 0
    for spec_id, spec_latent in image_mu.items():
        try:
            data.append({'spec_mu': spec_latent,
                        'spec_std': image_std[spec_id],
                        'lyrics_mu': spec_latent,  # dummy latent var
                        'lyrics_std': spec_latent,  # dummy latent var
                        'spec_id': spec_id})
        except KeyError as e:
            not_found += 1

    print(len(data))

    print(f'Total {not_found} missing specs')
    with open(f'data/processed/test_data.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    prepare_data()
