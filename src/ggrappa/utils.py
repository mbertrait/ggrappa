import torch
import numpy as np


def rss(data, axis=0):
    return np.sqrt(np.sum(np.abs(data)**2, axis=axis))


def get_indices_from_mask(mask):
    if not isinstance(mask, np.ndarray):
        mask = mask.numpy()
    nonzero_indices = np.nonzero(mask)

    min_indices = np.min(nonzero_indices, axis=1)
    max_indices = np.max(nonzero_indices, axis=1)

    cube_dimensions = max_indices - min_indices + 1

    return min_indices, cube_dimensions


def extract_acs(sig):
    _, ky, kz, _ = sig.shape

    start_ky = ky // 2
    start_kz = kz // 2

    left_start_ky = torch.max(torch.nonzero(sig[0, start_ky:, start_kz, 0], as_tuple=False)).item()
    left_end_ky = torch.max(torch.nonzero(sig[0, :start_ky+1, start_kz, 0].flip(0), as_tuple=False)).item()

    left_start_kz = torch.max(torch.nonzero(sig[0, start_ky, start_kz:, 0], as_tuple=False)).item()
    left_end_kz = torch.max(torch.nonzero(sig[0, start_ky, :start_kz+1, 0].flip(0), as_tuple=False)).item()

    return sig[:, start_ky-(left_start_ky+1):start_ky+left_end_ky, start_kz-(left_start_kz+1):start_kz+left_end_kz]
    

def pinv_batch(M, lambda_=1e-4, cuda=True):
    if cuda: M = M.cuda()
    MM = M.H @ M
    del M
    #torch.cuda.empty_cache()

    # Might also consider Power Iteration methods to speedup the process for large M matrix?
    S = torch.linalg.eigvalsh(MM)[-1].item()

    MM = MM.cpu()
    regularizer = (lambda_**2) * abs(S) * torch.eye(MM.shape[0], device=MM.device)
    del S
    #torch.cuda.empty_cache()
    reg_pinv = torch.linalg.pinv(MM + regularizer)
    del MM
    return reg_pinv

def pinv(M, lambda_=1e-4):
    if M.shape[0] > M.shape[1]:
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
    del A
    #torch.cuda.empty_cache()
    S = torch.linalg.eigvalsh(AA)[-1].item() # Largest eigenvalue
    lambda_sq = (lamdba_**2) * abs(S)
    del S
    #torch.cuda.empty_cache()
    I = torch.eye(AA.shape[0], dtype=AA.dtype, device=AA.device)
    regularized_matrix = AA + I * lambda_sq
    del I, AA, lambda_sq
    #torch.cuda.empty_cache()
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
