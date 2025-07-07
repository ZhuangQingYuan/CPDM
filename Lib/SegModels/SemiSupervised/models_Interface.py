from Lib.SegModels.SemiSupervised.Models.MTANET import MTAnet
from Lib.SegModels.SemiSupervised.Models.UNIMATCH import Unimatch
from Lib.SegModels.SemiSupervised.Models.CORRMATCH import CorrMatch
from Lib.SegModels.SemiSupervised.Models.SELFNET import SELFNET
from Lib.SegModels.SemiSupervised.Models.WSCL import WSCL
from Lib.SegModels.SemiSupervised.Models.LSST import LSST
from Lib.SegModels.SemiSupervised.Models.FIXMATCH import FIXMATCH
from Lib.SegModels.SemiSupervised.Models.UCCL import UCCL
from Lib.SegModels.SemiSupervised.Models.SELFNET_V1 import SELFNET_V1
from Lib.SegModels.SemiSupervised.Models.SCALEMATCH import SCALEMATCH

def SSegmentationSet(model: str, num_classes=5, img_size=256, **kwargs):
    if model =='mtanet':
        return MTAnet(kwargs['backbone'], num_classes=num_classes,upscale = kwargs['upscale'],mode = kwargs['mode'], disturbance_list = kwargs['disturbance_list'])
    elif model =='unimatch':
        return Unimatch(kwargs['cfg'])
    elif model == 'corrmatch':
        return CorrMatch(kwargs['cfg'])
    elif model == 'selfnet':
        return SELFNET(kwargs['cfg'])
    elif model == 'wscl':
        return WSCL(kwargs['cfg'])
    elif model == 'lsst':
        return LSST(kwargs['cfg'])
    elif model == 'fixmatch':
        return FIXMATCH(kwargs['cfg'])
    elif model == 'uccl':
        return UCCL(kwargs['cfg'])
    elif model == 'selfnet_v1':
        return SELFNET_V1(kwargs['cfg'])
    elif model == 'scalematch':
        return SCALEMATCH(kwargs['cfg'])
