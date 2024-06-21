import torch
import numpy as np
import nibabel as nib


def rss(data, axis=0):
    return np.sqrt(np.sum(np.abs(data)**2, axis=axis))


def pinv(M, lmda=1e-6):
    MM = M.H @ M
    S = torch.linalg.eigvalsh(MM)[-1].item()
    regularizer = (lmda**2) * abs(S) * torch.eye(MM.shape[0], device=M.device)
    reg_pinv = torch.linalg.pinv(MM + regularizer)
    return reg_pinv


def pinv_linalg(A, lmda=1e-2):
    AA = A.H@A
    S = torch.linalg.eigvalsh(AA)[-1].item() # Largest eigenvalue
    lambda_sq = (lmda**2) * abs(S)
    I = torch.eye(AA.shape[0], dtype=A.dtype, device=A.device)

    regularized_matrix = AA + I * lambda_sq

    return torch.linalg.solve(regularized_matrix, A.H)


def save_nifti(scan, filepath):
    scan_img = nib.Nifti1Image(scan, affine=np.eye(4))
    nib.save(scan_img, filepath)
