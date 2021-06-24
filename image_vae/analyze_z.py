'''import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open('image_z.pkl', 'rb') as f:
    image_z = pickle.load(f, encoding='bytes')

images = np.random.choice(list(image_z.keys()), 5)

for i in images:
    print(i)
    z = image_z[i]
    scored_z = []
    for k, v in image_z.items():
        scored_z.append((cosine_similarity(z, v)[0][0], k))
    scored_z = sorted(scored_z, reverse=True)
    print(scored_z[:10])'''

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open('image_mu_testing_ISMIR.pkl', 'rb') as f:
    image_z = pickle.load(f)

images = np.random.choice(list(image_z.keys()), 1)
#print(images)
#print(list(image_z.keys())[0])
#NineInchNails_came-back-haunted_20
#a = "DepecheMode_"
l = ['0.png']
#images = ['NineInchNails_gave-up_5.png']
#images = ["NineInchNails_no-you-dont_5.png"]
'''artists = ["NineInchNails_this-isnt-the-place_", "NineInchNails_gave-up_", "NineInchNails_shit-mirror_", "NineInchNails_lights-in-the-sky_", 
    "NineInchNails_every-day-is-exactly-the-same_", "NineInchNails_the-background-world_", "NineInchNails_were-in-this-together_",
    "NineInchNails_the-great-below_", "NineInchNails_no-you-dont_", "NineInchNails_im-looking-forward-to-joining-you-finally_", "NineInchNails_hurt_"]'''
'''artists = ["going-backwards_", "wrong_", "home_", "the-worst-crime_", "barrel-of-a-gun_",
    "sister-of-night_", "clean_", "useless_", "precious_", "walking-in-my-shoes_", 
    "never-let-me-down-again_", "its-no-good_"]'''
#artists = ["NineInchNails_various-methods-of-escape_"]
'''images=[]
for i in artists:
    for j in l:
        images.append(i+j)
print(images)'''
for i in images:
    print(i)
    try:
        z = image_z[i]
    except:
        continue
    scored_z = []
    for k, v in image_z.items():
        score = cosine_similarity(z, v)[0][0]
        if score != 1.0:
            scored_z.append((score, k))
    scored_z = sorted(scored_z, reverse=True)
    print(scored_z[:50])
    print('\n')