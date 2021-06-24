
from utils import *
from model.vae import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import model_config as config
from model.vae import *


def train(training_data):
    batch_size = config['batch_size']
    epochs = config['epochs']
    training_iterator = load_data(training_data, batch_size)
    vae = VAE().to(device)
    #model.load_state_dict(torch.load('vae.torch', map_location='cpu'))
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    for epoch in range(epochs):
        ind= 0
        print('Epoch: ' + str(epoch))
        for images in training_iterator:
            ind += 1
            images = images.to(device)
            recon_images, mu, logvar = vae(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(loss.item())
            if ind % 100 == 0:
                to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                                epochs, loss.item()/batch_size, bce.item()/batch_size, kld.item()/batch_size)
                print(to_print)

# notify to android when finished training
#notify(to_print, priority=1)

    torch.save(vae.state_dict(), config['model_name'])


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #BCE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def sample(x):
    x = torch.from_numpy(x)
    x = x.to(device)
    x = x.unsqueeze(0)
    print(x.shape)
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(config['model_name']))
    recon_x, _, _ = vae(x)
    print(x[0].shape)
    #save_image(x[0].data.cpu(), 'sample_image1.png')
    #save_image(recon_x[0].data.cpu(), 'sample_image2.png')
    compare_x = torch.cat([x, recon_x])
    print(compare_x.shape)
    save_image(compare_x.data.cpu(), config['image_name'])

def sample_z():
    '''with open('./data/ISMIR/ismir_latest_spec_array.pkl', 'rb') as f:
        concat = pickle.load(f, encoding='bytes')'''
    with open('./data/Instrumental/spec_array.pkl', 'rb') as f:
        concat = pickle.load(f, encoding='bytes')
    image_z = {}
    image_std = {}
    image_mu = {}
    aaa = np.array(list(concat.values()))
    print(aaa.shape)
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(config['model_name']))
    vae.eval()
    with torch.no_grad():
        for k,v in concat.items():
            x = np.array(v)
            #print(concat.shape)
            x = x/255.0
            x = x.astype('float32')
            x = x.transpose(2,0,1)
            x = torch.from_numpy(x)
            x = x.to(device)
            x = x.unsqueeze(0)
            #print(x.shape)
            z, mu, logvar = vae.encode(x)
            #print(z.shape)
            image_z[k] = z.cpu().numpy()
            std = logvar.mul(0.5).exp_()
            image_std[k] = std.cpu().numpy()
            image_mu[k] = mu.cpu().numpy()
            #save_image(x[0].data.cpu(), 'sample_image1.png')
            #save_image(recon_x[0].data.cpu(), 'sample_image2.png')
            #compare_x = torch.cat([x, recon_x])
            #print(compare_x.shape)
            #save_image(compare_x.data.cpu(), config['image_name'])
    '''with open('image_z_10sec_instrumental.pkl', 'wb') as f:
        pickle.dump(image_z, f)
    f.close()'''
    with open('image_std_testing_ISMIR.pkl', 'wb') as f:
        pickle.dump(image_std, f)
    f.close()
    with open('image_mu_testing_ISMIR.pkl', 'wb') as f:
        pickle.dump(image_mu, f)
    f.close()



