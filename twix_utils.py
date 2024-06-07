import mapvbvd
import torch


def read_twix_datafile(filepath, to_extract=["image", "refscan", "noise"], backend="torch"):
    scan = mapvbvd.mapVBVD(filepath)
    to_return = {}
    if "image" in to_extract:
        scan.image.squeeze = True
        scan.image.flagRemoveOS = True
        scan.image.flagDoAverage = True
        scan.image.flagIgnoreSeg = True
        img = scan.image['']
        if backend == "torch":
            sig = torch.from_numpy(img)
            sig = sig.permute(1,2,3,0)
        else:
            raise ValueError("Bad argument")
        to_return["image"] = img
    if "refscan" in to_extract:
        scan.refscan.squeeze = True
        scan.refscan.flagRemoveOS = True
        scan.refscan.flagDoAverage = True
        scan.refscan.flagIgnoreSeg = True
        ref = scan.refscan['']
        if backend == "torch":
            ref = torch.from_numpy(ref)
            ref = ref.permute(1,2,3,0)
        to_return["refscan"] = ref
    if "noise" in to_extract:
        scan.noise.squeeze = True
        scan.noise.flagRemoveOS = False
        scan.refscan.flagDoAverage = True
        scan.refscan.flagIgnoreSeg = True
        noise = scan.noise['']
        if backend == "torch":
            noise = torch.from_numpy(noise)
        to_return['noise'] = noise

    
    return to_return, scan