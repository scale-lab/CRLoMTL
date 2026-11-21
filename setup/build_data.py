from data.build_mtl_dataset import create_nyud_loader, create_pascal_loader, create_celeba_loader


DATASETS_LOADER = {
    "NYUD": create_nyud_loader,
    "PASCALContext": create_pascal_loader,
    "CelebA": create_celeba_loader,

}

def build_dataloaders(config):
    dataset = config.DATA.DATASET
    return DATASETS_LOADER[dataset](config)