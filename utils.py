import torch
import numpy as np
import nibabel as nib


def rss(data, axis=0):
    return np.sqrt(np.sum(np.abs(data)**2, axis=axis))


def extract_acs(sig):
    _, ky, kz, _ = sig.shape
    
    start_ky = ky // 2
    start_kz = kz // 2

    left_start_ky = np.max(sig[0,start_ky:,0,0].nonzero())
    left_end_ky = np.max(sig[0,:start_ky,0,0][::-1].nonzero())

    left_start_kz = np.max(sig[0,0,start_kz:,0].nonzero())
    left_end_kz = np.max(sig[0,0,:start_kz,0][::-1].nonzero())

    return sig[:, left_start_ky:left_end_ky+1, left_start_kz:left_end_kz+1]
    

def pinv_batch(M, lambda_=1e-4, cuda=True):
    if cuda: M = M.cuda()
    MM = M.H @ M
    del M
    torch.cuda.empty_cache()

    # Might also consider Power Iteration methods to speedup the process for large M matrix?
    S = torch.linalg.eigvalsh(MM)[-1].item()

    MM = MM.cpu()
    regularizer = (lambda_**2) * abs(S) * torch.eye(MM.shape[0], device=MM.device)
    del S
    torch.cuda.empty_cache()
    reg_pinv = torch.linalg.pinv(MM + regularizer)
    del MM
    return reg_pinv

def pinv(M, lambda_=1e-4):
    if M.shape[0] >= M.shape[1]:
        MM = M.H @ M
        finalTranspose = False
    else:
        MM = M @ M.H
        finalTranspose = True
    S = torch.linalg.eigvalsh(MM)[-1].item()
    regularizer = (lambda_**2) * abs(S) * torch.eye(MM.shape[0], device=M.device)
    reg_pinv = torch.linalg.pinv(MM + regularizer)
    return reg_pinv.H if finalTranspose else reg_pinv


def pinv_linalg_batch(A, lamdba_=1e-4, cuda=True):
    if cuda: A = A.cuda()
    AA = A.H@A
    #del A
    #torch.cuda.empty_cache()
    S = torch.linalg.eigvalsh(AA)[-1].item() # Largest eigenvalue
    lambda_sq = (lamdba_**2) * abs(S)
    del S
    torch.cuda.empty_cache()
    I = torch.eye(AA.shape[0], dtype=AA.dtype, device=AA.device)
    regularized_matrix = AA + I * lambda_sq
    del I, AA, lambda_sq
    torch.cuda.empty_cache()
    A = A.cpu()
    regularized_matrix = regularized_matrix.cpu()
    return torch.linalg.solve(regularized_matrix, A.H)


def pinv_linalg(A, lamdba_=1e-4):
    m,n = A.shape
    if n > m:
        AA = A@A.H
    else:
        AA = A.H@A
    S = torch.linalg.eigvalsh(AA)[-1].item()
    lambda_sq = (lamdba_**2) * abs(S)
    I = torch.eye(AA.shape[0], dtype=A.dtype, device=A.device)

    regularized_matrix = AA + I * lambda_sq

    return torch.linalg.solve(regularized_matrix, A.H)



def save_nifti(scan, filepath):
    scan_img = nib.Nifti1Image(scan, affine=np.eye(4))
    nib.save(scan_img, filepath)


# def compute_g_factor(undersampled_kspace, acs, grappa_weights, noise_correlation_matrix, reconstructed_data):
#     """
#     Compute the g-factor map for a multi-coil MRI using GRAPPA.

#     Parameters:
#     - undersampled_kspace: A numpy array of shape (Nc, Nx, Ny), the undersampled k-space data.
#     - acs: A numpy array of shape (Nc, ACSx, ACSy), the autocalibration signal.
#     - grappa_weights: A numpy array of shape (Nc, Nc, ky, kx), the GRAPPA weights.
#     - noise_correlation_matrix: A numpy array of shape (Nc, Nc), the noise correlation matrix of the coils.
#     - reconstructed_data: A numpy array of shape (Nc, Nx, Ny), the reconstructed data using GRAPPA.

#     Returns:
#     - g_factor_map: A numpy array of shape (Nx, Ny), the g-factor map.
#     """
#     Nc, Nx, Ny = undersampled_kspace.shape
    
#     # Inverse of the noise correlation matrix
#     noise_inv = np.linalg.inv(noise_correlation_matrix)

#     # Initialize the g-factor map
#     g_factor_map = np.zeros((Nx, Ny))
    
#     # Compute coil sensitivity maps from ACS data
#     coil_sensitivities = np.fft.ifftn(acs, axes=(1, 2))
#     coil_sensitivities = np.abs(coil_sensitivities)
#     coil_sensitivities /= np.max(coil_sensitivities, axis=(1, 2), keepdims=True)
    
#     # Loop over each pixel to compute the g-factor
#     for x in range(Nx):
#         for y in range(Ny):
#             # Extract the coil sensitivities at the current pixel
#             s = coil_sensitivities[:, x, y]
            
#             # Compute the noise covariance matrix
#             R = np.zeros((Nc, Nc))
#             for i in range(Nc):
#                 for j in range(Nc):
#                     R[i, j] = np.dot(grappa_weights[i, :, y, x], grappa_weights[j, :, y, x].T)
            
#             # Multiply with the noise correlation matrix
#             R = np.dot(np.dot(s.T, noise_inv), s) * R

#             # Compute the g-factor at the current pixel
#             g_factor_map[x, y] = np.sqrt(np.trace(R))
    
#     return g_factor_map
