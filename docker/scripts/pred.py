import os

import nibabel as nib
import numpy as np
from skimage.transform import resize

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import models

img_extended = namedtuple('img_extended',('img','seg','k','t','coord','cid'))

def load_volume_abdom(source_file):
    
        nimg = nib.load(source_file)
        nimg_array = nimg.get_fdata()
        vol_s = nimg_array.shape
        
        nimg_array = resize(nimg_array[:,:,::16], (160, 160, 32))
        nimg_array = nimg_array.transpose((2,1,0))
        nimg_array = nimg_array[:, ::-1, :]
        nimg_array = nimg_array * 2 - 1
        
        coord = np.linspace(-.5,.5,nimg_array.shape[0])[:, np.newaxis]
        
        img_batch = img_extended(nimg_array,
                                   np.zeros(32,dtype='uint8'),
                                   np.zeros(32,dtype='uint8'),
                                   np.zeros(32,dtype='uint8'),
                                   coord,
                                   np.zeros(32,dtype='uint8'),
                                  )
        
        return img_batch, vol_s, nimg.affine
    
    
    
def load_volume_brain(source_file):
        
        nimg = nib.load(source_file)
        nimg_array = nimg.get_fdata()
        vol_s = nimg_array.shape

        nimg_array = resize(nimg_array[:,:,::8], (160, 160, 32))
        nimg_array = nimg_array.transpose((2,1,0))
        nimg_array = nimg_array[:, ::-1, :]
        
        # Normalize non-zeros
        nimg_array[nimg_array<0.05] = 0

        non_zero_mask = np.where(nimg_array!=0)
        mu,sigma = nimg_array[non_zero_mask].mean(),nimg_array[non_zero_mask].std()
        nimg_array[non_zero_mask] = (nimg_array[non_zero_mask] - mu) / (sigma+1e-5)
        
        coord = np.linspace(-.5,.5,32)[:, np.newaxis]
        
        img_batch = img_extended(nimg_array,
                                   np.zeros(32,dtype='uint8'),
                                   np.zeros(32,dtype='uint8'),
                                   np.zeros(32,dtype='uint8'),
                                   coord,
                                   np.zeros(32,dtype='uint8'),
                                  )
        
        return img_batch, vol_s, nimg.affine


def resize_prediction(scores, source_shape):
    """Reverses original transformations and returns a new array equivalent to original volume"""
    scores = scores[:, ::-1, :]
    scores = scores.transpose((2,1,0))
    scores = resize(scores, source_shape)

    return scores

    
def reconstruct(n, img, cond = None, threshold_log_p = 5):
    """ Generates n reconstructions for each image in img.
    Resamples latent variables with cross-entropy > threshold
    Returns corrected images and associated latent variables"""
          
    #Use VQ-VAE to encode original image
    codes = model.retrieve_codes(img,cond)
    code_size = codes.shape[-2:]
    
    with torch.no_grad():

        samples = codes.clone().unsqueeze(1).repeat(1,n,1,1).reshape(img.shape[0]*n,*code_size)
        cond_repeat = cond.unsqueeze(1).repeat(1,n,1).reshape(img.shape[0]*n,-1)
        
        for r in range(code_size[0]):
            for c in range(code_size[1]):

                logits = model.forward_latent(samples, cond_repeat)[:, :, r, c]
                loss = F.cross_entropy(logits, samples[:, r, c], reduction='none')

                probs = F.softmax(logits, dim=1)
                samples[loss > threshold_log_p, r, c] = torch.multinomial(probs, 1).squeeze(-1)[loss > threshold_log_p]
        
        z = feat_ext_mdl.codebook.embedding(samples.unsqueeze(1))
        z = z.squeeze(1).permute(0,3,1,2).contiguous()
        
        # Split the calculation in batches
        x_tilde = []
        for i in range(img.shape[0]):
            x_tilde.append(feat_ext_mdl.decode(z[i*n:(i+1)*n],
                                               cond_repeat[i*n:(i+1)*n]))
        x_tilde = torch.cat(x_tilde)
        
        
    return x_tilde.reshape(img.shape[0],n,*img.shape[-2:]), samples.reshape(img.shape[0],n,*code_size)

        
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)
    parser.add_argument("-d", type=str, default="brain", help="can be either 'brain' or 'abdom'.", required=False)
    parser.add_argument("--no_gpu", type=bool, default=False, help="Do not use gpu", required=False)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode
    dataset = args.d
    no_gpu = args.no_gpu
    
    if no_gpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    
    # DATASET SPECIFIC SETTINGS
    if dataset == "abdom":
        parameters = {"threshold_sample": 7,
                     "threshold_pixel_correct": 5,
                     "checkpoint_features":'/workspace/checkpoints/abdom_feats.pt',
                     "checkpoint_latent": '/workspace/checkpoints/abdom_la.pt',
                     "load_function":load_volume_abdom,
                     "vq_net":{"d":1,"n_channels":(16,32,64,128,256),"code_size":128,"n_res_block":2,"cond_channels":1, 
                          "categorical":False, "reconstruction_loss":F.l1_loss},
                     "ar_net":{"shape":(10,10), "n_block":4, "n_res_block":4, "n_channels":128, "cond_channels":1,
                              "ini_kernel_size":3, "downsample_attn":1}}
    elif dataset == "brain":
        parameters = {"threshold_sample": 7,
                     "threshold_pixel_correct": 5,
                     "checkpoint_features":'/workspace/checkpoints/brain_feats.pt',
                     "checkpoint_latent": '/workspace/checkpoints/brain_la.pt',
                     "load_function":load_volume_brain,
                     "vq_net":{"d":1,"n_channels":(16,32,64,256),"code_size":128,"n_res_block":2,"cond_channels":1, 
                          "categorical":False, "reconstruction_loss":F.l1_loss},
                     "ar_net":{"shape":(20,20), "n_block":4, "n_res_block":4, "n_channels":128, "cond_channels":1}}
    
    
    # INITIALIZE SMOOTHING OF PIXEL PREDICTIONS
    smooth = nn.Sequential(nn.MaxPool3d(kernel_size=3,padding=1,stride=1),
                           nn.AvgPool3d(kernel_size=(3,7,7),padding=(1,3,3),stride=1),
                          )
    smooth.to(device)
        
    # INITIALIZE VQ MODEL
    feat_ext_mdl = models.VQVAE(**parameters["vq_net"])
    feat_ext_mdl.to(device).eval()
    
    chpt = torch.load(parameters["checkpoint_features"])
    feat_ext_mdl.load_state_dict(chpt)

    
    # INITIALIZE LATENT MODEL
    model = models.VQLatentSNAIL(feature_extractor_model=feat_ext_mdl,**parameters["ar_net"])
    model.to(device).eval()
    
    chpt = torch.load(parameters["checkpoint_latent"])
    model.load_state_dict(chpt)
        
    
    # ITERATE THROUGH FILES IN FOLDER
    for f in os.listdir(input_dir):

        source_file = os.path.join(input_dir, f)
        
        # LOAD FILE
        img_batch, vol_s, affine = parameters["load_function"](source_file)
            
        x_img = torch.from_numpy(img_batch.img.copy()).to(device).float()
        x_coord = torch.from_numpy(img_batch.coord.copy()).to(device).float()
        
        # IF SAMPLE, USE AS PREDICTOR NUMBER OF LATENT VARIABLES WITH LOSS > 7
        if mode == "sample":
            with torch.no_grad():
                loss = model.loss(x_img, cond = x_coord, reduction='none')["loss"].flatten(1)
                
                # As score, use the number of latent variable with log-likelihood > threshold 
                score = torch.sum(loss * (loss > parameters["threshold_sample"]),1).float()
                score = score.sum() / 2000
                score = score.clamp_max(1.).cpu().numpy() 
                        
            with open(os.path.join(output_dir, f + ".txt"), "w") as write_file:
                write_file.write(str(score))
                
        # IF PIXEL, USE DIFFERENCE WITH NEAREST RECONSTRUCITON AS SCORE
        elif mode == "pixel":
            x_tilde, z_tilde = reconstruct(15, x_img, x_coord, threshold_log_p = parameters["threshold_pixel_correct"])
            
            # Calculate absolute pixel wise difference 
            with torch.no_grad():
                score = torch.abs(x_img.unsqueeze(1)-x_tilde)
                
                sim_imgwise = torch.mean(score,(2,3)).unsqueeze(2).unsqueeze(3)
                sim_imgwise = torch.softmax(3/sim_imgwise,1)
                score = (score*sim_imgwise).sum(1,keepdims=True)

#                 score = score.mean(1,keepdims=True)
#                 near_neighbour = torch.sum(score,(2,3)).argmin(1)
#                 score = torch.gather(score,1,near_neighbour.view(-1,1,1,1).expand(-1,-1,160,160))                
                
                # Smooth output with 3d filter
                score = score.squeeze().unsqueeze(0).unsqueeze(0)
                score = -smooth(-score)
                
                # Normalize using 1 as max value
                score = score / 1.
                score = score.clamp_max(1.)
                                
            score = score.squeeze().cpu().numpy()
            score = resize_prediction(score,vol_s)
            
            final_nimg = nib.Nifti1Image(score, affine=affine)
            nib.save(final_nimg, os.path.join(output_dir, f))
           
