import torch
import math

from tqdm import tqdm
from typing import Any, Union, Tuple, List


def fixShapeAndIFFT(sig):
    sig_r = torch.zeros(32,400,400,320, dtype=torch.complex64)
    sig_r[:, :, 50:350, 1:320] = sig

    for nc in range(sig.shape[0]):
        sig_r[nc] = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifft(torch.fft.ifft(torch.fft.fftshift(sig_r[nc], dim=(0,1,2)), dim=0), dim=1), dim=2), dim=(0,1,2))

    return sig_r


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


# def pinv(M, lmda=1e-6):
#     regularizer = lmda * torch.linalg.norm(M) * torch.eye(M.shape[0])
#     reg_pinv = torch.linalg.pinv(M @ M.conj().T + regularizer)
#     return reg_pinv

def pinv_linalg(A, lmda=1e-2, cuda=False, lanczos=True):
    #TODO: Let user select backeng (numpy, scipy, torch (cuda or not))
    if cuda:
        A = A.cuda()
    AA = A.conj().T@A
    S = torch.linalg.eigvalsh(AA)[-1].item() # Largest eigenvalue
    lambda_sq = (lmda**2) * abs(S)
    I = torch.eye(AA.shape[0], dtype=A.dtype)

    regularized_matrix = AA + I * lambda_sq

    A_pinv = torch.linalg.solve(regularized_matrix, A.conj().T)
    return A_pinv.cpu() if cuda else A_pinv



def grappa3D(
        sig: torch.Tensor,
        acs: torch.Tensor,
        af: Union[list[int], tuple[int, ...]],
        block_size=None,
        grappa_kernel=None,
        cuda=True,
        verbose=True
) -> torch.Tensor:
    
    nc, ny, nz, nx = sig.shape
    acsny, acsnz, acsnx = acs.shape[1:]

    total_af = math.prod(af)

    #TODO: Might need to do checks on arguments passed to the function

    # Generate the acceleration pattern
    #TODO: Will need to think about `block_size` argument that can be given by the user
    pat = torch.zeros(*(3*a+1 for a in af))
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
    idxs_srcT = (pat[:sbly, :sblz].T == 1).flatten()

    idxs_tgs = torch.zeros(sbly, sblz, sblx)
    idxs_tgs[ypos:ypos+tbly, zpos:zpos+tblz, xpos:xpos+1] = 1
    idxs_tgs = (idxs_tgs == 1)

    nsp = idxs_src.sum()

    if grappa_kernel is None:

        ########################################################
        #                  Kernel estimation                   #
        ########################################################
        if verbose:
            print("Kernel estimation...")


        windows = acs.unfold(dimension=1, size=sbly, step=1)
        windows = windows.unfold(dimension=2, size=sblz, step=1)
        windows = windows.unfold(dimension=3, size=sblx, step=1)
        windows = windows.permute(-1,-2,-3,0,1,2,3)

        src = windows[idxs_src.permute(2,1,0)].reshape(nc*nsp, -1)
        tgs = windows[idxs_tgs.permute(2,1,0)].reshape(nc*tbly*tblz*tblx, -1)
        
        if cuda:
            grappa_kernel = tgs @ pinv_linalg(src.T, cuda=cuda).T

    ########################################################
    #                  Kernel application                  #
    ########################################################

    grappa_kernel = reshape_fortran(grappa_kernel, (nc * tbly * tblz, nc * nsp // sblx, sblx))
    grappa_conv = torch.zeros(nc * tbly * tblz, nc * nsp // sblx, nx, dtype=sig.dtype, device=sig.device)
    start_idx = nx // 2 - sblx // 2
    end_idx = start_idx + sblx

    grappa_conv[:, :, start_idx:end_idx] = torch.flip(grappa_kernel, [2])
    
    if sblx>1:
        sig = torch.fft.ifftshift(torch.fft.ifft(torch.fft.fftshift(sig, dim=-1), dim=-1), dim=-1)

    if nx > 1:
        grappa_conv = torch.fft.ifftshift(torch.fft.ifft(torch.fft.fftshift(grappa_conv, dim=-1), dim=-1), dim=-1)
        sig = torch.cat((sig[:, -af:, :, :], sig, sig[:, :af, :, :]), dim=1)
    
    if nz > 1:
        sig = torch.cat((sig[:, :, -2*af:, :], sig, sig[:, :, :2*af, :]), dim=2)

    shift_y=2
    sig_tmp = torch.zeros_like(sig).cuda()
    np = int(sum(idxs_srcT))

    y_ival = range(shift_y, ny+2-af-sbly, afy)
    z_ival = range(0, nz+4*af-sblz, afz)
    grappa_conv = grappa_conv.cuda()

    for y in tqdm(y_ival):
        acq_y = sig[:, y:y+sbly, :, :].cuda()
        for z in z_ival:
            acq = reshape_fortran(acq_y[:,:,z:z+sblz, :], (nc, sbly*sblz, nx))
            acq = reshape_fortran(acq[:, idxs_srcT, :], (nc*np,1, nx))
            sig_tmp[:,y+ypos:y+ypos+tbly, z+zpos:z+zpos+tblz, :] =  reshape_fortran(
                                                                        torch.bmm(
                                                                            grappa_conv.permute(2,0,1),
                                                                            acq.permute(2,0,1)
                                                                        ).permute(1,2,0),
                                                                        (nc, tbly, tblz, nx)
                                                                    )

    if cuda:
        sig_tmp = sig_tmp.cpu()
    return sig_tmp
