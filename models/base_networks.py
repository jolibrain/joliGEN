from .modules.fid.pytorch_fid.inception import InceptionV3


def define_inception(device, dims):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    return model
