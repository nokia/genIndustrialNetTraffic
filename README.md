# A Generative Approach for Production-Aware Industrial Network Traffic Modeling
Pretrained generated models for production-aware industrial network traffic modeling

## Contribution
we investigate the network traffic data generated from a laser cutting machine deployed in a Trumpf factory in Germany. We evaluate the data and model the network traffic as a production state dependent stochastic process in two steps: first, we model the production process as a multi-state semi Markov process, then we learn the conditional distributions of the production state dependent packet interarrival time and packet size with different generative models, including variational autoencoder (VAE), conditional variational autoencoder (CVAE), and generative adversarial network (GAN).

In this repository, we publish the pretrained models for the industrial network traffic. 

![](Distribution_sojourn_time.png)

## Dependencies
We have used the following software versions:
- python (3.8)
  - numpy (1.22)
  - scikit-learn (1.0.2)
  - pandas (1.3.5)
  - pytorch (1.10.1)
  
## Launching scripts

The most easy way to launch the scripts is to run `tox -e run` in the `generative_models` directory in a new conda environment. 

Dependencies are collected to the [requirements.txt](generative_models\requirements.txt) file. 
The traffic generation can be launched through the `main.py` script. 

`saved_model*` directories are the pre-trained models that are used for generating the distributions. They are loaded in
the main.py for generating our distributions and there is no need to change or modify them.

## Modifying the configuration

This can be achieved by chanding the code of the script. 

* `model` variable in line 25 selects the model used, must be VAE, CVAE or GAN
* `x_dim` variable in line 29 selects the dimension of the output. 1 means only distribution of interarrival time
  between packets, 2 means joint distribution of interarrival time and packet size
* Line 85 is the output, which is a dictionary which has as a key the industrial state and as value the list of interarrival time and/or packet size

## Example output

Running the script should result in similar diagrams than [1D_distribution_CVAE.png](generative_models\1D_distribution_CVAE.png), 
[1D_distribution_GAN.png](generative_models\1D_distribution_GAN.png) or [1D_distribution_VAE.png](generative_models\1D_distribution_VAE.png).

## License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
© 2022 Nokia
