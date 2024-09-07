import torch
import math

from .utils import pinv, pinv_linalg
from tqdm import tqdm
from typing import Union


def GRAPPA_Recon(
        sig: torch.Tensor,
        acs: torch.Tensor,
        af: Union[list[int], tuple[int, ...]],
        delta: int = 0,
        kernel_size: Union[list[int], tuple[int, ...]] = (4,4,5),
        lambda_=1e-4,
        batch_size: int = 1,
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
    delta : int
        For CAIPIRINHA undersampling pattern. Default: `0`.
    kernel_size : Union[list[int], tuple[int, ...]]
        GRAPPA kernel size
    lambda_ : float
        Regularization parameter of the pseudo-inverse.
    grappa_kernel : torch.Tensor, optional
        GRAPPA kernel to be used. If `None`, the GRAPPA kernel weights will be computed. Default: `None`.
    cuda : bool, optional
        Whether to use GPU or not. Default: `True`.
    verbose : bool, optional
        Activate verbose mode (printing) or not. Default: `True`.
    """
    #TODO: Check support of 2D-CAIPIRINHA undersmapling pattern 
    #TODO: If the acs is not provided: extract it from sig (trivial)
    if len(af) == 1:
        af = [af[0], 1]

    if len(sig.shape) == 3: # 2D multicoil
        sig = sig[:, :, None, :]
        if len(af) == 2:
            af = [af[0], 1]

    if len(acs.shape) == 3:
        acs = acs[:, :, None, :]

    nc, ny, nz, nx = sig.shape
    acsny, acsnz, acsnx = acs.shape[1:]

    if verbose:
        print("GRAPPA Kernel size: ", kernel_size)
        print("lambda: ", lambda_)
        print("batch size: ", batch_size)

    if kernel_size:
        pat = torch.zeros([((k-1) * af[i] + 1) for i, k in enumerate(kernel_size[:2])])
    else:
        pat = torch.zeros((3*total_af if af[0] > 1 else 1, 3*total_af if af[1] > 1 else 1))

    slices = [slice(None, None, a) for a in af]
    pat[tuple(slices)] = 1

    total_af = math.prod(af)

    #TODO: Might need to do checks on arguments passed to the function

    # Generate the acceleration pattern
    #TODO: Will need to think about `block_size` argument that can be given by the user

    # Determine the target block size from total_af
    tbly = af[0]
    tblz = af[1]
    tblx = 1

    # Determine 3D source block size
    #sblx = min(7, acsnx)
    #sbly = min(max(total_af, 3*tbly), acsny)
    #sblz = min(max(total_af, 3*tblz), acsnz)

    sbly = min(pat.shape[0], acsny)
    sblz = min(pat.shape[1], acsnz)
    sblx = min(kernel_size[-1], acsnx)

    xpos = (sblx-1)//2
    ypos = (sbly-tbly)//2
    zpos = (sblz-tblz)//2

    idxs_src = (pat[:sbly, :sblz] == 1)
    idxs_src = idxs_src.unsqueeze(-1).expand(*idxs_src.size(), sblx)

    idxs_tgs = torch.zeros(sbly, sblz, sblx)
    idxs_tgs[ypos:ypos+tbly, zpos:zpos+tblz, xpos:xpos+1] = 1
    idxs_tgs = (idxs_tgs == 1)
    #idxs_tgs = (idxs_tgs & ~idxs_src)
    #idxs_tgs[idxs_src == 1] = 0

    nsp = idxs_src.sum()

    #if grappa_kernel is None:
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
        tgs = blocks[..., idxs_tgs.flatten()].reshape(nc, -1, idxs_tgs.sum())

        src = src.permute(1,0,-1).reshape(-1, nc*nsp)
        tgs = tgs.permute(1,0,-1).reshape(-1, nc*idxs_tgs.sum())

        src = src.cuda() if cuda else src
        tgs = tgs.cuda() if cuda else tgs

        grappa_kernel = pinv(src, lambda_) @ src.H @ tgs
        #grappa_kernel = (tgs.T @ pinv_linalg(src).T).T
        del src, tgs, blocks
        torch.cuda.empty_cache()

    ########################################################
    #                  Kernel application                  #
    ########################################################

    shift_y = (abs(sig[0,:,0,0]) > 0).nonzero()[0].item()
    shift_z = (abs(sig[0,shift_y,:,0]) > 0).nonzero()[0].item()


    sig = torch.nn.functional.pad(sig,  (xpos, (sblx-xpos-tblx),
                                        (af[1] - zpos)%tblz + zpos, (sblz-zpos-tblz),
                                        (af[0] - ypos)%tbly + ypos, (sbly-ypos-tbly)))

    rec = torch.zeros_like(sig)

    size_chunk_y = sbly + tbly*(batch_size - 1)
    y_ival = range(shift_y, max(rec.shape[1] - sbly, 1), tbly*batch_size)
    z_ival = range(shift_z, max(rec.shape[2] - sblz, 1), tblz)

    if verbose:
        print("GRAPPA Reconstruction...")
    
    idxs_src = idxs_src.flatten()

    for y in tqdm(y_ival, disable=not verbose):
        sig_y = sig[:,y:y+size_chunk_y]
        if cuda:
            sig_y = sig_y.cuda()
        for z in tqdm(z_ival, disable=not verbose):
            blocks = sig_y[:,:, z:z+sblz, :].unfold(dimension=1, size=sbly, step=tbly).unfold(dimension=3, size=sblx, step=tblx)
            blocks = blocks.permute(1,3,0,4,2,5)
            cur_batch_sz_y = blocks.shape[0]
            cur_batch_sz_x = blocks.shape[1]
            blocks = blocks.reshape(cur_batch_sz_y, cur_batch_sz_x, nc, -1)[..., idxs_src]
            rec[:, y+ypos:y+ypos+tbly*cur_batch_sz_y, z+zpos:z+zpos+tblz, xpos:xpos+tblx*cur_batch_sz_x] =  (blocks.reshape(cur_batch_sz_y*cur_batch_sz_x, -1) @ grappa_kernel) \
                                                                                                            .reshape(cur_batch_sz_y, cur_batch_sz_x, nc, tbly, tblz, tblx) \
                                                                                                            .permute(2,0,3,4,1,5) \
                                                                                                            .reshape(nc, cur_batch_sz_y*tbly, tblz, cur_batch_sz_x*tblx)

        del sig_y
        if cuda: torch.cuda.empty_cache()
    
    if sbly > 1:
        rec = rec[:, (af[0] - ypos)%tbly + ypos:-(sbly-ypos-tbly)]
    
    if sblz > 1:
        rec = rec[...,(af[1] - zpos)%tblz + zpos:-(sblz-zpos-tblz),:]

    if sblx > 1:
        rec = rec[...,xpos:-(sblx-xpos-tblx)]

    return rec
