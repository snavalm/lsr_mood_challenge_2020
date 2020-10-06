# Latent Space Restoration for Anomaly Detection with Vector Quantised VAE

 We propose an out-of-distribution detection method that combines density and reconstruction-based approaches based on a vector-quantised variational auto-encoder (VQ-VAE) model. VQ-VAE model learns to encode images in a categorical latent space. The prior distribution of latent codes is then modelled using an autoregressive (AR) model, in our case with PixelSNAIL architecture. We found that the prior density estimated by the AR model can be useful for unsupervised anomaly detection and consequently we derived both sample and pixel-wise scores from it.  Sample-wise score is defined as the negative log-likelihood of the latent variables, considering only latent variables above a threshold that defines highly unlikely codes. Additionally, with the objective of localising anomalies, out-of-distribution images are restored into in-distribution images by replacing unlikely latent codes with samples from the prior model and decoding to pixel space. The average of L1 distance between generated restorations and original image is used as pixel-wise anomaly score. 
 
 Please use jupyter notebooks to get started:
- [0DatasetExtraction-MOODBrainDataset.ipynb](0DatasetExtraction-MOODBrainDataset.ipynb) used to unzip original MOOD zip dataset and generate files as expected by dataloaders
- [1LatentSpatialRestoration-MOODBrainDataset.ipynb](1LatentSpatialRestoration-MOODBrainDataset.ipynb) defines architectures, training procedures and anomaly scores. This notebook only includes brain dataset, abdominal dataset submission is identical with only the following differences:
  * Pre-processing does not standardize pixel intensities. It also does not exclude the first and last 20 axial slices from the original volume.
  * VQ-VAE includes one extra block (16,32,64,**128**,256) so latent becomes (10,10) instead of (20,20)
- [Docker](docker) submitted in mood challenge 

