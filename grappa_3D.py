import torch
import torch.nn as nn
from tqdm import tqdm


def pinv(M, lmda=0.01):
    regularizer = lmda * torch.linalg.norm(M) * torch.eye(M.shape[0])
    reg_pinv = torch.linalg.pinv(M @ M.conj().T + regularizer)
    return reg_pinv


def grappa3D(sig, acs, afy, afz, window_size=None, grappa_kernel=None, cuda=False):
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

    if grappa_kernel is None:


        ########################################################
        #                  Kernel estimation                   #
        ########################################################


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
        torch.save(grappa_kernel, "grappa_k.torch")

    ########################################################
    #                  Kernel application                  #
    ########################################################

    # Zero-Pad the signal
    sig = nn.functional.pad(sig, (sblz//2 - tblz//2, sblz//2 - tblz//2, sbly//2 - tbly//2, sbly//2 - tbly//2, sblx//2, sblz//2))
    nx_pad, ny_pad, nz_pad = sig.shape[1:]

    n_shift_x = nx_pad - sblx + 1
    n_shift_y = ny_pad - sbly + 1
    n_shift_z = nz_pad - sblz + 1
    """
    for shift_x in tqdm(range(n_shift_x)):
        for shift_y in range(0, n_shift_y, tbly):
            for shift_z in range(0, n_shift_z, tblz):
                    s = sig[:, shift_x:shift_x + sblx,
                               shift_y:shift_y + sbly,
                               shift_z:shift_z + sblz].reshape(nc, -1)
                    s = s[:, idxs_src].reshape(-1)
                    t = sig[:, shift_x + xpos:shift_x + xpos + 1,
                               shift_y + ypos:shift_y + ypos + tbly,
                               shift_z + zpos:shift_z + zpos + tblz].reshape(-1)
                    
                    sig[:, shift_x + xpos:shift_x + xpos + 1,
                           shift_y + ypos:shift_y + ypos + tbly,
                           shift_z + zpos:shift_z + zpos + tblz]  = (grappa_kernel @ s.T).squeeze().reshape(nc, 1, tbly, tblz)
    """
    #n_shift_y, n_shift_z = sig.shape[2], sig.shape[3]  # assuming sig is a 4D tensor with shape (nc, ..., ..., ...)
    n_shift_x = nx_pad - sblx + 1
    n_shift_y = ny_pad - sbly + 1 - acsny
    n_shift_z = nz_pad - sblz + 1 - acsnz
    n_blocks = n_shift_x * (n_shift_y // tbly) * (n_shift_z // tblz)

    # Initialize the large S and T matrice
    S = torch.zeros(nc * nsp, n_blocks, dtype=sig_dtype)
    T = torch.zeros(nc * 1 * tbly * tblz, n_blocks, dtype=sig_dtype)

    #sig = sig.cuda()
    #S = S.cuda()
    #T = T.cuda()

    block_idx = 0
    print("making blocks")
    for shift_x in tqdm(range(n_shift_x)):
        for shift_y in range(0, n_shift_y, tbly):
            for shift_z in range(0, n_shift_z, tblz):
                if not (ny_pad//2 - acsny//2 <= shift_y <= ny_pad//2 + acsny//2 and nz_pad//2 - acsnz//2 <= shift_z <= nz_pad//2 + acsnz//2):
                    s = sig[:, shift_x:shift_x + sblx,
                            shift_y:shift_y + sbly,
                            shift_z:shift_z + sblz].reshape(nc, -1)
                    s = s[:, idxs_src].reshape(-1)
                    t = sig[:, shift_x + xpos:shift_x + xpos + 1,
                            shift_y + ypos:shift_y + ypos + tbly,
                            shift_z + zpos:shift_z + zpos + tblz].reshape(-1)

                    S[:, block_idx] = s
                    T[:, block_idx] = t
                    block_idx += 1

    if cuda:
        grappa_kernel = grappa_kernel.cuda()
        S = S.cuda()
        #sig = sig.cuda()
    # Perform the giant matrix multiplication
    result = grappa_kernel @ S.T
    print("done")
    result = result.cpu()

    sig = sig.cuda()
    # Place the result back into sig
    block_idx = 0
    for shift_x in tqdm(range(n_shift_x)):
        for shift_y in tqdm(range(0, n_shift_y, tbly)):
            for shift_z in range(0, n_shift_z, tblz):
                result_block = result[:, block_idx].reshape(nc, 1, tbly, tblz)
                sig[:, shift_x + xpos:shift_x + xpos + 1,
                    shift_y + ypos:shift_y + ypos + tbly,
                    shift_z + zpos:shift_z + zpos + tblz] = result_block
                block_idx += 1




# import mapvbvd
# scan = mapvbvd.mapVBVD(r"/home/mb279022/meas_MID00231_FID25576_t2_spc_sag_iso0_55mm_p3x2.dat")
# scan.image.squeeze = True
# scan.image.flagRemoveOS = True
# scan.image.flagDoAverage = True
# scan.image.flagIgnoreSeg = True
# sig = torch.from_numpy(scan.image[''])
# sig = sig.permute(1,0,2,3)

# scan.refscan.squeeze = True
# scan.refscan.flagRemoveOS = True
# scan.refscan.flagDoAverage = True
# scan.refscan.flagIgnoreSeg = True
# ref = torch.from_numpy(scan.refscan[''])
# ref = ref.permute(1,0,2,3)

# grappa3D(sig, ref, 3, 2, grappa_kernel=torch.load("grappa_k.torch"))
