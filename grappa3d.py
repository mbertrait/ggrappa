import numpy as np
from numpy.linalg import pinv
import gc

def grappa3d_conv_pe4(sig, acs, afy, afz=1, delta=0, kcenter=None, ws_conv=None, sbl=None):
    # Get the dimensions of the input data
    nc, nx, ny, nz = sig.shape

    af = afy * afz  # Calculate total reduction factor

    # Create the 2D CAIPIRINHA-type undersampled sampling pattern
    pat = np.zeros((3*af+1, 3*af+1), dtype=int)
    cnt = 0
    for k in range(0, af, afy):
        pat[k::af, cnt::afz] = 1
        cnt = (cnt + delta) % afz

    # Determine target block size
    tbly = int(af / np.sum(pat[0:af, 0]))
    tblz = int(af / np.sum(pat[0, 0:af]))
    if tbly >= tblz:
        tbly = af // tblz
    else:
        tblz = af // tbly

    # Determine 3D source block size
    if ws_conv is None:
        sbl = [min(7, acs.shape[1]), int(min(max(af, 3*tbly), acs.shape[2])), min(max(af, 3*tblz), acs.shape[3])]
    elif sbl is None:
        raise ValueError('source size (sbl) required')
    sblx, sbly, sblz = sbl

    # Get matrix indices of the acquired data in 3D source block
    pat = pat[0:sbly, 0:sblz]
    idx = pat.ravel() == 1

    # Predefine the position of the 2D target block in the 3D source block
    ypos = (sbly - tbly) // 2
    zpos = (sblz - tblz) // 2

    if ws_conv is None:
        xpos = (sblx - 1) // 2
        idx = np.repeat(idx, sblx)

        nsp = np.sum(idx)

        # Calculation of the GRAPPA weights
        ncacs, nxacs, nyacs, nzacs = acs.shape
        if ncacs != nc:
            raise ValueError('number of coils in data & acs does not match')

        # Collect all the source and target replicates within the ACS data
        src = np.zeros((nc*nsp, (nyacs-sbly+1)*(nzacs-sblz+1)*(nxacs-sblx+1)), dtype=sig.dtype)
        trg = np.zeros((nc*tbly*tblz, (nyacs-sbly+1)*(nzacs-sblz+1)*(nxacs-sblx+1)), dtype=sig.dtype)

        cnt = 0
        for z in range(nzacs-sblz+1):
            for y in range(nyacs-sbly+1):
                for x in range(nxacs-sblx+1):
                    cnt += 1

                    # Source points
                    s = acs[:, x:x+sblx, y:y+sbly, z:z+sblz]
                    s = s.reshape(nc, -1)
                    s = s[:, idx]

                    # Target points
                    t = acs[:, x+xpos, y+ypos:y+ypos+tbly, z+zpos:z+zpos+tblz]
                    t = t.reshape(nc, -1)

                    if np.any(np.sum(s, axis=1) == 0) or np.any(np.sum(t, axis=1) == 0):
                        # Check whether there is missing acs data
                        continue

                    src[:, cnt-1] = s.ravel()
                    trg[:, cnt-1] = t.ravel()

        # Solve for the weights using pseudo inverse
        p_inv = pinv_reg(src)
        print("pinv valculated")
        ws = trg @ p_inv.conj().T

        del trg, src
        gc.collect()
        # Convolution of weights in the read-direction (x)
        ws = ws.reshape((sblx, nc*tbly*tblz, nc*nsp//sblx))
        ws_conv = np.zeros((nx, nc*tbly*tblz, nc*nsp//sblx), dtype=sig.dtype)
        ws_conv[(nx//2)-sblx//2:(nx//2)-sblx//2+sblx, :, :] = np.flip(ws, axis=0)

        if nx > 1:
            ws_conv = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(ws_conv, axes=0), axis=0), axes=0)
        ws_conv = np.transpose(ws_conv, (1, 2, 0))  # nx -> last dim
        del ws
        gc.collect()

    # Reconstruction
    if kcenter is not None and len(kcenter) > 0:
        if delta == 0:
            shift_y = (kcenter[0] - 1) % afy
            shift_z = (kcenter[1] - 1) % afz
        else:
            tmp = np.zeros(kcenter, dtype=bool)
            for y in range(kcenter[0], 0, -afy):
                shift = (delta * (kcenter[0] - y) // afy) % afz
                tmp[y, kcenter[1]-shift::-afz] = True

            shift_z = np.argmax(np.sum(tmp, axis=0))  # Select first z-partition with acquired data
            shift_y = np.argmax(tmp[:, shift_z])  # Select first measured y-line within this partition
    else:
        tmp = np.sum(np.sum(np.abs(sig), axis=1), axis=0)
        shift_y = 0
        shift_z = 0
        if delta > 0:
            for k in range(afy):
                if tmp[k, 0] > 0:
                    shift_y = k
                    break
            for k in range(afz):
                if tmp[0, k] > 0:
                    shift_z = k
                    break
        else:
            tmp_z = np.sum(tmp, axis=0)
            tmp_y = np.sum(tmp, axis=1)
            for k in range(afy):
                if tmp_y[k] > 0:
                    shift_y = k
                    break
            for k in range(afz):
                if tmp_z[k] > 0:
                    shift_z = k
                    break

    # Fourier Transformation in x (Convolution!)
    if sblx > 1:
        sig = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(sig, axes=1), axis=1), axes=1)
        idx = idx[0::sblx]

    # Slightly increase size of sig to include edges of k-space
    if ny > 1:
        sig = np.concatenate((sig[:, :, -af+1:], sig, sig[:, :, :af]), axis=2)
    if nz > 1:
        sig = np.concatenate((sig[:, :, :, -2*af+1:], sig, sig[:, :, :, :2*af]), axis=3)

    if ny == 1:
        y_ival = [0]
    else:
        y_ival = range(shift_y, ny+2*af-sbly+1, afy)

    if nz == 1:
        z_ival = [0]
    else:
        z_ival = range(0, nz+4*af-sblz+1, afz)

    y_sz = sig.shape[2]
    z_sz = sig.shape[3]
    sig_tmp = np.zeros((nc, y_sz, z_sz, nx), dtype=sig.dtype)
    np_ = np.sum(idx)

    for y in tqdm(y_ival):
        acq_y = sig[:, :, y:y+sbly, :]
        for z in z_ival:
            acq = acq_y[:, :, :, z:z+sblz]
            acq = acq.reshape(nc, nx, -1)
            acq = acq[..., idx]
            acq = acq.reshape((nc*np_.item(), 1, nx))

            tmpp = np.einsum('ijk,jlk->ilk', ws_conv, acq)

            sig_tmp[:, y+ypos:y+ypos+tbly, z+zpos:z+zpos+tblz, :] = np.reshape(tmpp, (nc, tbly, tblz, nx))

    sig = np.transpose(sig_tmp, (0, 3, 1, 2))
    if ny > 1:
        sig = sig[:, :, af:(ny+af), :]
    if nz > 1:
        sig = sig[:, :, :, 2*af-1:(nz+2*af-1)]

    recon = None
    if nargout > 2:
        recon = nx * np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(sig, axes=(2, 3)), axes=(2, 3)), axes=(2, 3))

    if sblx > 1:
        sig = nx * np.fft.fftshift(np.fft.fft(np.fft.fftshift(sig, axes=1), axis=1), axes=1)

    return sig, ws_conv, sbl, recon


def pinv_reg(A, lambda_=1e-2):
    print("pinv")
    if A.size == 0:
        return np.zeros_like(A.T)

    m, n = A.shape
    if m < n:
        
        #X = pinv(A@A + lambda_ * np.eye(n)) @ A.T
        X = np.linalg.solve(A@A.conj().T + np.eye(m)*lambda_, A)
    else:
        X = A @ pinv(A @ A + lambda_ * np.eye(m))
    return X
