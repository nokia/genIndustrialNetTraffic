import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import utils_NN
import seaborn as sns
import pickle

def convert_dict_into_distribution(temp_dict, columns_name):
    ls_df = []
    for key, value in temp_dict.items():
        sub_df = pd.DataFrame(columns=columns_name)
        sub_df[columns_name[0]] = np.array(value)
        sub_df[columns_name[1]] = key
        ls_df.append(sub_df)
        
    df = pd.concat(ls_df)
    df = df.reset_index(drop=True)
    return df

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(100)
torch.manual_seed(100)
list_states = ['Running', 'Reentry', 'Stopped', 'Aborted', 'Ended']
model = 'CVAE'
number_generated_samples = 5000
norm_factor_max = 18.27
norm_factor_min = 0.0
x_dim = 1           ## 1 dim --> interarrival time, 2 dim --> (intearrival time, packet size)
recon_dist = {}
num_samples = number_generated_samples

assert x_dim==1 or x_dim==2, "Input dimension is not valid. Only 1 or 2 is currently supported."
assert model=='VAE' or model =='CVAE' or model=='GAN', "Input model not available. Try one amongst VAE, CVAE or GAN"

if model=='CVAE':
    with open("saved_model_allStates/cvae_"+str(x_dim)+"D_interarrival_configuration.pkl", 'rb') as handle:
        cfg = pickle.load(handle)
    latent_dim = cfg.model.latent_dim
    encoder_layersizes = cfg.model.encoder_layersizes
    decoder_layersizes = cfg.model.decoder_layersizes
    y_dim = len(list_states)
    vae = utils_NN.VAE(x_dim, y_dim, latent_dim, encoder_layersizes, decoder_layersizes, cfg.sample.normparams, conditional=True).to(device)
    vae.load_state_dict(torch.load("saved_model_allStates/cvae_"+str(x_dim)+"D_interarrival.pth"))
    vae.eval()
    labels = {'Running': 0, 'Reentry': 1, 'Stopped': 2, 'Aborted': 3, 'Ended': 4}

for state in list_states:
    if model=='CVAE':
        z = torch.randn([num_samples, latent_dim]).to(device)
        y = [labels[state]]*num_samples
        y = torch.LongTensor(y).to(device)
        recon_dist[state] = vae.inference(z,y).detach().numpy()

    if model=='GAN':        ## GAN generating samples
        with open("saved_model_"+state+"/GAN_"+str(x_dim)+"D_configuration.pkl", 'rb') as handle:
            cfg = pickle.load(handle)
        latent_dim = cfg.model.latent_dim
        generator_layersizes = cfg.model.encoder_layersizes
        discriminator_layersizes = cfg.model.decoder_layersizes
        gan = utils_NN.GAN(x_dim, latent_dim, generator_layersizes, discriminator_layersizes).to(device)
        gan.load_state_dict(torch.load("saved_model_"+state+"/gan_"+str(x_dim)+"D.pth"))
        gan.eval()
        generator = gan.initialize_generator(device)
        generator.load_state_dict(torch.load("saved_model_"+state+"/generator_"+str(x_dim)+"D.pth"))
        generator.eval()
        recon_dist[state] = gan.generate_samples(generator, num_samples, latent_dim, device)

    elif model=='VAE':
        with open("saved_model_"+state+"/vae_"+str(x_dim)+"D_interarrival_configuration.pkl", 'rb') as handle:
            cfg = pickle.load(handle)
        latent_dim = cfg.model.latent_dim
        encoder_layersizes = cfg.model.encoder_layersizes
        decoder_layersizes = cfg.model.decoder_layersizes
        y_dim = cfg.sample.normparams.shape[0]
        vae = utils_NN.VAE(x_dim, y_dim, latent_dim, encoder_layersizes, decoder_layersizes, cfg.sample.normparams, conditional=False).to(device)
        vae.load_state_dict(torch.load("saved_model_"+state+"/vae_"+str(x_dim)+"D_interarrival.pth"))
        vae.eval()
        ''' generate samples '''
        z = torch.randn([num_samples, latent_dim]).to(device)
        recon_dist[state] = vae.inference(z).detach().numpy()
    
      ## the samples are logscaled and normalized. We need to get the Denormalization factors
    recon_dist[state] = recon_dist[state]*(norm_factor_max - norm_factor_min) + norm_factor_min
    recon_dist[state] = np.exp(recon_dist[state].reshape(-1))-1

if x_dim==1:
    ''' compare distributions of actions and recon_x '''
    recon_packet_interarrival = convert_dict_into_distribution(recon_dist, ['Interarrival time [ms]', 'state'])
    recon_packet_interarrival['Interarrival time [ms]'] = np.log10(recon_packet_interarrival['Interarrival time [ms]']+1)
    n_bins=100
    sns.set_style("whitegrid")
    f, axes = plt.subplots(1, 1)
    sns.histplot(recon_packet_interarrival, x='Interarrival time [ms]', hue='state', stat='probability', kde=True, ax=axes, bins=n_bins, legend=True).set_title('Generated distribution with '+model)
    axes.set_xticks([1, 3, 5, 7])
    axes.set_xticklabels([r'$10^1$',r'$10^3$', r'$10^5$',r'$10^7$'])
    plt.tight_layout()
    # plt.show()
    plt.savefig('1D_distribution_'+model+'.png')
