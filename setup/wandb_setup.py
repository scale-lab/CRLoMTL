import os


try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False


def setup_wandb(config, tag):
    if not config.WANDB:
        return

    assert HAS_WANDB, "wandb not installed"

    if not os.getenv("WANDB_API_KEY"):
        wandb.login()
    else:
        wandb.login(key=os.getenv("WANDB_API_KEY"))

    wandb.init(project="CRLoMTL", name=tag, config=config)
    wandb.config.update(config)