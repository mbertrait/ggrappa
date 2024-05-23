import torch
import torch.nn as nn
from tqdm.auto import tqdm


def pinv(M, lmda=0.01):
    regularizer = lmda * torch.linalg.norm(M) * torch.eye(M.shape[0])
    reg_pinv = torch.linalg.pinv(M @ M.conj().T + regularizer)
    return reg_pinv


def grappa3D(sig, acs, afy, afz, window_size=None):
    nc, nx, ny, nz = sig.shape
    acsnc, acsnx, acsny, acsnz = acs.shape
    sig_dtype = sig.dtype
    af = afy*afz

    pat = torch.zeros(3*af+1, 3*af+1) if not window_size else torch.zeros(*window_size)
    pat[::afy, ::afz] = 1

    # Determine the target block size
    tbly = int(af/sum(pat[:af, 0]))
    tblz = int(af/sum(pat[0, :af]))

    # Determine 3D source block size
    sblx = min(5, acsnx)
    sbly = min(max(af, 3*tbly), acsny)
    sblz = min(max(af, 3*tblz), acsnz)

    xpos = (sblx-1)//2
    ypos = (sbly-tbly)//2
    zpos = (sblz-tblz)//2

    # indexes of sources points in the window
    idxs_src = pat[:sbly, :sblz] == 1
    #idxs_src = idxs_src.unsqueeze(0).repeat(sblx,1,1).reshape(-1)

    idxs_src = torch.stack([idxs_src] * sblx).reshape(-1)
    nsp = idxs_src.sum()

    n_shift_x = acsnx - sblx + 1
    n_shift_y = acsny - sbly + 1
    n_shift_z = acsnz - sblz + 1

    src = torch.zeros(nc*nsp, n_shift_x*n_shift_y*n_shift_z, dtype=sig_dtype)
    tgs = torch.zeros(nc*tbly*tblz, n_shift_x*n_shift_y*n_shift_z, dtype=sig_dtype)

    for shift_x in tqdm(range(n_shift_x)):
        for shift_y in range(n_shift_y):
            for shift_z in range(n_shift_z):

                s = acs[:, shift_x:shift_x + sblx,
                           shift_y:shift_y + sbly,
                           shift_z:shift_z + sblz].reshape(nc, -1)
                s = s[:, idxs_src].reshape(-1)

                t = acs[:, shift_x + xpos:shift_x + xpos + 1,
                           shift_y + ypos:shift_y + ypos + tbly,
                           shift_z + zpos:shift_z + zpos + tblz].reshape(-1)
                                
                src[...,shift_x+shift_y+shift_z] = s
                tgs[...,shift_x+shift_y+shift_z] = t

    
    grappa_kernel = tgs @ src.conj().T @ pinv(src)
