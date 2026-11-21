def unpack_batch(batch, config):

    return (

        batch

        if config.DATA.DATASET == "toy"

        else (batch["image"], {t: batch[t] for t in config.TASKS})

    )