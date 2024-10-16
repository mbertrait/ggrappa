import torch
import logging

from typing import Union, Tuple
from tqdm import tqdm

from . import GRAPPAReconSpec
from .utils import extract_sampled_regions, get_indices_from_mask, pad_back_to_size

logger = logger = logging.getLogger(__name__)


def apply_grappa_kernel(sig,
                        grappa_recon_spec : GRAPPAReconSpec,
                        *,
                        batch_size: int = 1,
                        isGolfSparks: bool = False,
                        mask = None,
                        cuda: bool = False,
                        cuda_mode: str = "all",
                        return_kernel: bool = False,
                        quiet: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    grappa_kernel = grappa_recon_spec.weights.cuda() if cuda and cuda_mode in ["all", "application"] else grappa_recon_spec.weights
    xpos, ypos, zpos = grappa_recon_spec.pos
    tbly, tblz, tblx = grappa_recon_spec.tbl
    sbly, sblz, sblx = grappa_recon_spec.sbl
    af = grappa_recon_spec.af
    delta = grappa_recon_spec.delta

    nc, *vol_shape = sig.shape

    if mask is not None:
        sig_ = sig
        left, size = get_indices_from_mask(mask)
        sig = (sig*mask)[...,left[0]:left[0]+size[0],
                             left[1]:left[1]+size[1],
                             left[2]:left[2]+size[2]]
        
    shift_y, shift_z = 0, 0
    if isGolfSparks:
        sig, start_loc, end_loc = extract_sampled_regions(sig, acs_only=False)
        samples_axis = [
            sig.abs().sum(0).sum(0).sum(-1),
            sig.abs().sum(0).sum(1).sum(-1)
        ]
        shifts = [
            torch.argmax(torch.stack([
                torch.sum(samples_axis[axis][i::af_axis]) 
                for i in range(af_axis)
            ])).item()
            for axis, af_axis in enumerate(af)
        ]
        shift_y, shift_z = shifts
    else:
        shift_y, shift_z = abs(sig).sum(0).sum(-1).nonzero()[0]
        sig = torch.nn.functional.pad(sig,  (xpos, (sblx-xpos-tblx),
                                            (af[1] - zpos)%tblz + zpos, (sblz-zpos-tblz),
                                            (af[0] * (1 if delta == 0 else af[1]//delta) - ypos)%tbly + ypos, (sbly-ypos-tbly)))

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
            are_targets_fully_sampled = (blocks.abs().sum(2)!=0).sum(-1) == idxs_src.sum()
            if isGolfSparks:
                if not torch.any(are_targets_fully_sampled):
                    # If we have no blocks with all samples, skip this iteration
                    continue    
                locs_fully_sampled = torch.nonzero(are_targets_fully_sampled, as_tuple=True)
                res = torch.zeros((cur_batch_sz_y, cur_batch_sz_x, nc, tbly, tblz, tblx), dtype=blocks.dtype, device=blocks.device)
                blocks = blocks[locs_fully_sampled[0], locs_fully_sampled[1]]
                blocks = blocks.reshape(blocks.shape[0], -1)
                res[locs_fully_sampled[0], locs_fully_sampled[1]] = (blocks @ grappa_kernel).reshape(len(locs_fully_sampled[0]), nc, tbly, tblz, tblx)
            else:
                blocks = blocks.reshape(cur_batch_sz_y*cur_batch_sz_x, -1)
                res = (blocks @ grappa_kernel).reshape(cur_batch_sz_y, cur_batch_sz_x, nc, tbly, tblz, tblx)
            
            rec[:,  y+ypos:y+ypos+tbly*cur_batch_sz_y,
                    z+zpos:z+zpos+tblz,
                    xpos:xpos+tblx*cur_batch_sz_x]  =   res.permute(2,0,3,4,1,5).reshape(nc, cur_batch_sz_y*tbly, tblz, cur_batch_sz_x*tblx)

        del sig_y
        if cuda: 
            torch.cuda.empty_cache()

    rec[abs(sig) > 0] = sig[abs(sig) > 0]
    
    if isGolfSparks:
        rec = pad_back_to_size(rec, vol_shape, start_loc, end_loc)
    else:
        # Remove the padding
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
        logger.info("GRAPPA Reconstruction done.")

    return rec, grappa_kernel if return_kernel else rec