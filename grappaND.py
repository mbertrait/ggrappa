import torch
import logging

from tqdm import tqdm
from typing import Union

from .utils import pinv, extract_acs, get_indices_from_mask


logger = logging.getLogger(__name__)


def GRAPPA_Recon(
        sig: torch.Tensor,
        acs: torch.Tensor,
        af: Union[list[int], tuple[int, ...]],
        delta: int = 0,
        kernel_size: Union[list[int], tuple[int, ...]] = (4,4,5),
        lambda_: float = 1e-4,
        batch_size: int = 1,
        grappa_kernel: torch.Tensor = None,
        mask: torch.Tensor = None,
        cuda: bool = True,
        cuda_mode: str = "all",
        quiet=False,
) -> torch.Tensor:
    """Perform GRAPPA reconstruction.

    Parameters
    ----------
    sig : torch.Tensor
        Complex 4D Tensor of shape: (nc, ky, kz, kx).
    acs : torch.Tensor
        Complex 4D Tensor of shape: (nc, acsky, acskz, acskx).
    af : Union[list[int], tuple[int, ...]]
        Acceleration factors. [afy, afz].
    delta : int, optional
        CAIPIRINHA shift. Default: `0`.
    kernel_size : Union[list[int], tuple[int, ...]], optional
        GRAPPA kernel size. Default `(4,4,5)`
    lambda_ : float, optional
        Regularization parameter of the pseudo-inverse. Default: `1e-4`
    batch_size : int, optional
        Size of the batch of `windows` to process by iteration in the kernel application phase. Default: `1`.
    grappa_kernel : torch.Tensor, optional
        GRAPPA kernel to be used. If `None`, the GRAPPA kernel weights will be computed. Default: `None`.
    mask : torch.Tensor, optional
        Binary mask for masked kernel application. Shape: (ky, kz, kx). Default: `None`
    cuda : bool, optional
        Whether to use GPU or not. Default: `True`.
    cuda_mode : str, optional
        CUDA operation mode (GPU):
            * "all" - Both kernel estimation and kernel application . Memory intensive.
            * "estimation" - Only use CUDA for GRAPPA kernel estimation.
            * "application" - Only use CUDA for GRAPPA kernel application.
        Default: `all`.
    quiet : bool, optional
        Enable printings and tqdm bars. Default: `True`.
    """

    if len(af) == 1:
        af = [af[0], 1]

    if len(sig.shape) == 3: # 2D multicoil
        sig = sig[:, :, None, :]
        if len(af) == 2:
            af = [af[0], 1]
    
    if grappa_kernel is None:
        if acs is None:
            acs = extract_acs(sig)

        if len(acs.shape) == 3:
            acs = acs[:, :, None, :]
    
    if mask is not None:
        assert sig.shape[1:] == mask.shape

    nc = sig.shape[0]
    acsny, acsnz, acsnx = acs.shape[1:]

    logger.debug("GRAPPA Kernel size: ", kernel_size)
    logger.debug("lambda: ", lambda_)
    logger.debug("batch size: ", batch_size)

    if kernel_size:
        pat = torch.zeros([((k-1) * af[i]*[1, (af[1]//delta)][i==0] + 1) for i, k in enumerate(kernel_size[:2])])

    cnt=0
    for y in range(0, pat.shape[0], af[0]):
        pat[y,cnt::af[1]] = 1
        cnt = (cnt+delta)%af[1]

    tbly = af[0]*(af[1]//delta)
    tblz = af[1]
    tblx = 1

    sbly = min(pat.shape[0], acsny)
    sblz = min(pat.shape[1], acsnz)
    sblx = min(kernel_size[-1], acsnx)

    xpos = (sblx-1)//2
    ypos = (sbly-tbly)//2
    zpos = (sblz-tblz)//2

    idxs_src = (pat[:sbly, :sblz] == 1)
    idxs_src = idxs_src.unsqueeze(-1).expand(*idxs_src.size(), sblx)

    idxs_tgs = torch.zeros(sbly, sblz, sblx, dtype=torch.bool)
    idxs_tgs[ypos:ypos+tbly, zpos:zpos+tblz, xpos:xpos+1] = True

    nsp = idxs_src.sum()

    if grappa_kernel is None:

        ########################################################
        #                  Kernel estimation                   #
        ########################################################
        if not quiet:
            logger.info("GRAPPA Kernel estimation...")

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

        src = src.cuda() if cuda and cuda_mode in ["all", "estimation"] else src
        tgs = tgs.cuda() if cuda and cuda_mode in ["all", "estimation"] else tgs

        grappa_kernel = pinv(src, lambda_) @ src.H @ tgs

        del src, tgs, blocks
        torch.cuda.empty_cache()

    ########################################################
    #                  Kernel application                  #
    ########################################################

    grappa_kernel = grappa_kernel.cuda() if cuda and cuda_mode in ["all", "application"] else grappa_kernel

    if mask is not None:
        sig_ = sig
        left, size = get_indices_from_mask(mask)
        sig = (sig*mask)[...,left[0]:left[0]+size[0],
                             left[1]:left[1]+size[1],
                             left[2]:left[2]+size[2]]
        
    shift_y, shift_z = abs(sig).sum(0).sum(-1).nonzero()[0]

    sig = torch.nn.functional.pad(sig,  (xpos, (sblx-xpos-tblx),
                                        (af[1] - zpos)%tblz + zpos, (sblz-zpos-tblz),
                                        (af[0]*(af[1]//delta) - ypos)%tbly + ypos, (sbly-ypos-tbly)))

    rec = torch.zeros_like(sig)

    size_chunk_y = sbly + tbly*(batch_size - 1)
    y_ival = range(shift_y, max(rec.shape[1] - sbly, 1), tbly*batch_size)
    z_ival = range(shift_z, max(rec.shape[2] - sblz, 1), tblz)

    if not quiet:
        logger.info("GRAPPA Reconstruction...")
    
    idxs_src = idxs_src.flatten()

    for y in tqdm(y_ival, disable=quiet):
        sig_y = sig[:,y:y+size_chunk_y]
        sig_y = sig_y.cuda() if cuda and cuda_mode in ["all", "application"] else sig_y
        for z in z_ival:
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
    
    if mask is not None:
        rec *= mask[left[0]:left[0]+size[0],
                    left[1]:left[1]+size[1],
                    left[2]:left[2]+size[2]]

        not_mask = (~mask) if mask.dtype == torch.bool else (1 - mask)
        
        sig_[...,left[0]:left[0]+size[0],
                 left[1]:left[1]+size[1],
                 left[2]:left[2]+size[2]] = (sig_ * not_mask)[...,left[0]:left[0]+size[0],
                                                                  left[1]:left[1]+size[1],
                                                                  left[2]:left[2]+size[2]] + rec
        
        rec = sig_

    if not quiet:
        logger.info("GRAPPA Reconstruction...")

    return rec, grappa_kernel
