from pathlib import Path
import nibabel as nib
import numpy as np
from collections import namedtuple
import torch
import matplotlib.pyplot as plt

def train(model, train_loader, optimizer, device):
    model.train()
    train_losses = []
    batch_sizes = []
    for x in train_loader:
        img = torch.from_numpy(x.img)
        coord = torch.from_numpy(x.coord)
        loss = model.loss(img.to(device),coord.to(device))
        optimizer.zero_grad()

        loss['loss'].backward()

        # Gradient clippling
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)

        optimizer.step()
        train_losses.append(loss['loss'].item() * img.shape[0])
        batch_sizes.append(img.shape[0])

    return sum(train_losses)/sum(batch_sizes)

def eval_loss(model, data_loader, device):
    model.eval()
    eval_losses = []
    batch_sizes = []
    with torch.no_grad():
        for x in data_loader:
            img = torch.from_numpy(x.img)
            coord = torch.from_numpy(x.coord)
            loss = model.loss(img.to(device), coord.to(device))
            eval_losses.append(loss['loss'].item() * img.shape[0])
            batch_sizes.append(img.shape[0])

    return sum(eval_losses)/sum(batch_sizes)

def save_checkpoint(model,optimizer, tracker, file_name):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'tracker': tracker,
    }

    torch.save(checkpoint, file_name)

class train_tracker:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.lr = []

    def __len__(self):
        return len(self.train_losses)

    def append(self,train_loss,test_loss,lr):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.lr.append(lr)

    def plot(self,N=None):
        N = N if N is not None else self.__len__()
        plt.plot(self.train_losses[-N:],label='Train')
        plt.plot(self.test_losses[-N:], label='Eval')
        plt.legend()
        plt.show()

def train_epochs(model, optimizer,tracker, train_loader, test_loader, epochs, device, chpt = None):

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer,device)
        test_loss = eval_loss(model, test_loader, device)

        tracker.append(train_loss,test_loss,optimizer.param_groups[0]['lr'])

        print('{} epochs, {:.3f} test loss, {:.3f} train loss'.format(len(tracker), test_loss, train_loss))
        if chpt is not None:
            save_checkpoint(model,optimizer,tracker,
                            'checkpoints/{}_{:03}.pt'.format(chpt,len(tracker)))


def load_cid(cid,path):
    """Load segmentation and volume"""
    vol = nib.load(path+'/case_{:05d}/imaging.nii.gz'.format(cid))
    seg = nib.load(path+'/case_{:05d}/segmentation.nii.gz'.format(cid))
    spacing = vol.affine
    vol = np.asarray(vol.get_fdata())
    seg = np.asarray(seg.get_fdata())
    seg = seg.astype(np.int8)
    vol = normalize(vol)
    return vol, seg, spacing


img_extended = namedtuple('img_extended',('img','seg','k','t','coord','cid'))

def get_full_case_id(cid):
    try:
        cid = int(cid)
        case_id = "case_{:05d}".format(cid)
    except ValueError:
        case_id = cid

    return case_id


def get_case_path(cid):
    # Resolve location where data should be living
    data_path = Path(__file__).parent.parent / "data"
    if not data_path.exists():
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )

    # Get case_id from provided cid
    case_id = get_full_case_id(cid)

    # Make sure that case_id exists under the data_path
    case_path = data_path / case_id
    if not case_path.exists():
        raise ValueError(
            "Case could not be found \"{}\"".format(case_path.name)
        )

    return case_path

def dice_score(trues, preds):
    """Calculate dice score / f1 given binary boolean variables: 2 x IoU"""
    return 2. * (trues & preds).sum()/(trues.sum() + preds.sum())

def max_score(trues, pred, score_func = dice_score, steps = 8):
    """Iterate through possible threshold ranges and return max score and argmax threshold """
    min_d, max_d = pred.min(), pred.max()

    for i in range(steps):
        mid_d = (max_d-min_d)/2 + min_d
        mid_s = score_func(trues,pred > mid_d)

        q1_s = score_func(trues,pred > (max_d-min_d)/4 + min_d)
        q3_s = score_func(trues,pred > 3*(max_d-min_d)/4 + min_d)

        if q1_s == q3_s:
            break
        elif q1_s > q3_s:
            max_d = mid_d
        else:
            min_d = mid_d
    return mid_s, mid_d
