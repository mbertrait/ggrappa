import torch
import math

from utils import pinv, pinv_linalg
from tqdm import tqdm
from typing import Union


def GRAPPA_Recon(
        sig: torch.Tensor,
        acs: torch.Tensor,
        af: Union[list[int], tuple[int, ...]],
        grappa_kernel: torch.Tensor = None,
        cuda: bool = True,
        verbose: bool = True,
) -> torch.Tensor:
    """Perform GRAPPA reconstruction.

    For now only cartesian regular undersampling (no CAIPI) is supported.
    For now acs must be provided, in the future it will be deduced from sig.

    Parameters
    ----------
    sig : torch.Tensor
        Complex 4D Tensor of shape: (nc, ky, kz, kx) 
    acs : torch.Tensor
        Complex 4D Tensor of shape: (nc, acsky, acskz, acskx)
    af : Union[list[int], tuple[int, ...]]
        Acceleration factors. [afy, afz]
    grappa_kernel : torch.Tensor, optional
        GRAPPA kernel to be used. If `None`, the GRAPPA kernel weights will be computed. Default: `None`.
    cuda : bool, optional
        Whether to use GPU or not. Default: `True`.
    verbose : bool, optional
        Activate verbose mode (printing) or not. Default: `True`.
    """

    #TODO: Check support of 2D-CAIPIRINHA undersmapling pattern 
    #TODO: If the acs is not provided: extract it from sig (trivial)
    
    nc, ny, nz, nx = sig.shape
    acsny, acsnz, acsnx = acs.shape[1:]

    total_af = math.prod(af)

    # Generate the acceleration pattern
    #TODO: Will need to think about `block_size` argument that can be given by the user
    pat = torch.zeros((3*total_af,) * len(af))
    slices = [slice(None, None, a) for a in af]
    pat[tuple(slices)] = 1

    # Determine the target block size from total_af
    tbly = int(total_af/sum(pat[:total_af, 0]))
    tblz = int(total_af/sum(pat[0, :total_af]))
    tblx=1

    # Determine 3D source block size
    sblx = min(7, acsnx)
    sbly = min(max(total_af, 3*tbly), acsny)
    sblz = min(max(total_af, 3*tblz), acsnz)

    xpos = (sblx-1)//2
    ypos = (sbly-tbly)//2
    zpos = (sblz-tblz)//2

    idxs_src = (pat[:sbly, :sblz] == 1)
    idxs_src = idxs_src.unsqueeze(-1).expand(*idxs_src.size(), sblx)

    idxs_tgs = torch.zeros(sbly, sblz, sblx)
    idxs_tgs[ypos:ypos+tbly, zpos:zpos+tblz, xpos:xpos+1] = 1
    idxs_tgs = (idxs_tgs == 1)

    nsp = idxs_src.sum()

    if grappa_kernel is None:

        ########################################################
        #                  Kernel estimation                   #
        ########################################################
        if verbose:
            print("GRAPPA Kernel estimation...")

        # Keep an eye on torch.nn.Unfold (and its functionnal) for 3D sliding block support
        # for future replacement.
        # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html
        blocks = acs.unfold(dimension=1, size=sbly, step=1)
        blocks = blocks.unfold(dimension=2, size=sblz, step=1)
        blocks = blocks.unfold(dimension=3, size=sblx, step=1)
        blocks = blocks.flatten(start_dim=-3)

        src = blocks[..., idxs_src.flatten()].reshape(nc, -1, nsp)
        tgs = blocks[..., idxs_tgs.flatten()].reshape(nc, -1, tbly*tblz*tblx)

        src = src.permute(1,0,-1).reshape(-1, nc*nsp)
        tgs = tgs.permute(1,0,-1).reshape(-1, nc*tbly*tblz*tblx)

        src = src.cuda() if cuda else src
        tgs = tgs.cuda() if cuda else tgs

        #grappa_kernel = pinv(src) @ src.H @ tgs
        grappa_kernel = (tgs.T @ pinv_linalg(src).T).T
        del src, tgs, blocks

    ########################################################
    #                  Kernel application                  #
    ########################################################

    shift_y = (abs(sig[0,:,0,0]) > 0).nonzero()[0].item()
    shift_z = (abs(sig[0,shift_y,:,0]) > 0).nonzero()[0].item()

    if ny > 1:
        sig = torch.cat((sig[:, -total_af:, :, :], sig, sig[:, :total_af, :, :]), dim=1)
    
    if nz > 1:
        sig = torch.cat((sig[:, :, -2*total_af:, :], sig, sig[:, :, :2*total_af, :]), dim=2)

    if nx > 1:
        sig = torch.cat((sig[:, :, :,-total_af:], sig, sig[:, :, :, :total_af]), dim=3)

    rec = torch.zeros_like(sig)

    y_ival = range(shift_y, ny+2*total_af-sbly, af[0])
    z_ival = range(shift_z, nz+4*total_af-sblz, af[1])

    if verbose:
        print("GRAPPA Reconstruction...")

    idxs_src = idxs_src.flatten()

    for y in tqdm(y_ival):
        sig_y = sig[:, y:y+sbly]
        if cuda: sig_y = sig_y.cuda()
        for z in z_ival:
            blocks = sig_y[:,:, z:z+sblz, :].unfold(dimension=3, size=sblx, step=1)
            blocks = blocks.permute(3,0,1,2,4)
            cur_batch_sz = blocks.shape[0]
            blocks = blocks.reshape(cur_batch_sz, nc, -1)[..., idxs_src]
            rec[:, y+ypos:y+ypos+tbly, z+zpos:z+zpos+tblz, xpos:-xpos] = (blocks.reshape(cur_batch_sz, -1) @ grappa_kernel) \
                                                                         .reshape(cur_batch_sz, nc, tbly, tblz) \
                                                                         .permute(1,2,3,0)

    if cuda: rec = rec.cpu()
    rec[abs(sig) != 0] = sig[abs(sig) != 0] # Data consistency : some people do it others don't, need to check benefits or drawbacks on recon.
    rec = rec[:, total_af:ny+total_af, 2*total_af:nz+2*total_af,:]
    return rec
