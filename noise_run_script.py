"""Convenience entry points for running end-to-end noise experiments."""

from experiment_runner import RunConfig, mode_flags, run_training


def _run(flags, noise_percentage, epoch_weights=0, temperature=0, drop_k=0, backbone=0, loss=7, include_epoch_weights=True):
    """Run `view_classification_with_noise.py` once with the provided flags/config."""

    config = RunConfig(
        noise_percentage=noise_percentage,
        epoch_weights=epoch_weights if include_epoch_weights else None,
        temperature=temperature,
        drop_k=drop_k,
        backbone=backbone,
        loss=loss,
    )
    run_training(flags=flags, config=config)


def run_pretraining(noise_percentage, temperature=0, drop_k=0, backbone=0, loss=7):
    """Run pretraining only for a single noise setting."""

    _run(
        ["--pretraining"],
        noise_percentage,
        temperature=temperature,
        drop_k=drop_k,
        backbone=backbone,
        loss=loss,
        include_epoch_weights=False,
    )


def run_corrections(noise_percentage, epoch_weights=0, temperature=0, drop_k=0, backbone=0, loss=7):
    """Run label correction generation for a single noise setting."""

    _run(
        ["--correction"],
        noise_percentage,
        epoch_weights=epoch_weights,
        temperature=temperature,
        drop_k=drop_k,
        backbone=backbone,
        loss=loss,
    )


def run_plots(noise_percentage, epoch_weights=0, temperature=0, drop_k=0, backbone=0, loss=7):
    """Run plotting-only path for a single noise setting."""

    _run(
        [],
        noise_percentage,
        epoch_weights=epoch_weights,
        temperature=temperature,
        drop_k=drop_k,
        backbone=backbone,
        loss=loss,
    )


def run_downstream_fine_tuning_only(
    use_corrections, remove_as_noisy_labels, noise_percentage, epoch_weights=0, temperature=0, drop_k=0, backbone=0, loss=7
):
    """Run downstream fine-tuning from pretrained weights."""

    flags = ["--downstream_training", "--downstream_with_pretrainedweights", *mode_flags(use_corrections, remove_as_noisy_labels)]
    _run(
        flags,
        noise_percentage,
        epoch_weights=epoch_weights,
        temperature=temperature,
        drop_k=drop_k,
        backbone=backbone,
        loss=loss,
    )


def run_downstream_fine_tuning_linear_eval(
    use_corrections, remove_as_noisy_labels, noise_percentage, epoch_weights=0, temperature=0, drop_k=0, backbone=0, loss=7
):
    """Run linear-evaluation mode for downstream training."""

    flags = ["--downstream_training", "--downstream_with_linear", *mode_flags(use_corrections, remove_as_noisy_labels)]
    _run(
        flags,
        noise_percentage,
        epoch_weights=epoch_weights,
        temperature=temperature,
        drop_k=drop_k,
        backbone=backbone,
        loss=loss,
    )


def run_downstream_imagenet(
    use_corrections, remove_as_noisy_labels, noise_percentage, epoch_weights=0, temperature=0, drop_k=0, backbone=0, loss=7
):
    """Run downstream training initialized from ImageNet weights."""

    flags = ["--downstream_with_imagenet", "--downstream_training", *mode_flags(use_corrections, remove_as_noisy_labels)]
    _run(
        flags,
        noise_percentage,
        epoch_weights=epoch_weights,
        temperature=temperature,
        drop_k=drop_k,
        backbone=backbone,
        loss=loss,
    )


def run_downstream_random_init(
    use_corrections, remove_as_noisy_labels, noise_percentage, epoch_weights=0, temperature=0, drop_k=0, backbone=0, loss=7
):
    """Run downstream training from random initialization."""

    flags = ["--downstream_with_rand_init", "--downstream_training", *mode_flags(use_corrections, remove_as_noisy_labels)]
    _run(
        flags,
        noise_percentage,
        epoch_weights=epoch_weights,
        temperature=temperature,
        drop_k=drop_k,
        backbone=backbone,
        loss=loss,
    )


def run_downstream(use_corrections, remove_as_noisy_labels, noise_percentage, epoch_weights=0, temperature=0, drop_k=0, backbone=0, loss=7):
    """Run the full downstream protocol with ImageNet and pretrained-weight options."""

    flags = [
        "--downstream_with_imagenet",
        "--downstream_training",
        "--downstream_with_pretrainedweights",
        *mode_flags(use_corrections, remove_as_noisy_labels),
    ]
    _run(
        flags,
        noise_percentage,
        epoch_weights=epoch_weights,
        temperature=temperature,
        drop_k=drop_k,
        backbone=backbone,
        loss=loss,
    )


def run_inference_only(
    use_corrections, remove_as_noisy_labels, noise_percentage, epoch_weights=0, temperature=0, drop_k=0, backbone=0, loss=7
):
    """Run inference-only evaluation with downstream checkpoints."""

    flags = ["--downstream_with_imagenet", "--downstream_with_pretrainedweights", *mode_flags(use_corrections, remove_as_noisy_labels)]
    _run(
        flags,
        noise_percentage,
        epoch_weights=epoch_weights,
        temperature=temperature,
        drop_k=drop_k,
        backbone=backbone,
        loss=loss,
    )


def run_inference_set(noise_percentage, epoch_weights, temperature=0, drop_k=0, backbone=0, loss=7):
    """Run inference for baseline, corrected-label, and removed-label variants."""

    run_inference_only(False, False, noise_percentage, epoch_weights, temperature, drop_k, backbone, loss)
    run_inference_only(True, False, noise_percentage, epoch_weights, temperature, drop_k, backbone, loss)
    run_inference_only(False, True, noise_percentage, epoch_weights, temperature, drop_k, backbone, loss)


def run_downstream_set_full(noise_percentage, epoch_weights, temperature=0, drop_k=0, backbone=0, loss=7):
    """Run the full downstream set across correction/removal variants."""

    run_downstream(False, False, noise_percentage, epoch_weights, temperature, drop_k, backbone, loss)
    run_downstream(True, False, noise_percentage, epoch_weights, temperature, drop_k, backbone, loss)
    run_downstream(False, True, noise_percentage, epoch_weights, temperature, drop_k, backbone, loss)


def run_downstream_set_random_init(noise_percentage, epoch_weights, temperature=0, drop_k=0, backbone=0, loss=7):
    """Run random-initialization downstream baseline for one noise setting."""

    run_downstream_random_init(False, False, noise_percentage, epoch_weights, temperature, drop_k, backbone, loss)


def run_downstream_set_imagenet(noise_percentage, epoch_weights, temperature=0, drop_k=0, backbone=0, loss=7):
    """Run ImageNet-initialized downstream baseline for one noise setting."""

    run_downstream_imagenet(False, False, noise_percentage, epoch_weights, temperature, drop_k, backbone, loss)


def run_downstream_finetuning_set(noise_percentage, epoch_weights, temperature=0, drop_k=0, backbone=0, loss=7):
    """Run pretrained-weight fine-tuning baseline for one noise setting."""

    run_downstream_fine_tuning_only(False, False, noise_percentage, epoch_weights, temperature, drop_k, backbone, loss)


def run_downstream_linear_set(noise_percentage, epoch_weights, temperature=0, drop_k=0, backbone=0, loss=7):
    """Run linear-evaluation baseline for one noise setting."""

    run_downstream_fine_tuning_linear_eval(False, False, noise_percentage, epoch_weights, temperature, drop_k, backbone, loss)


def main():
    """Execute the default experiment schedule used in this repository."""

    temperature = 0.1
    drop_k = 1
    backbone = 5
    loss = 0  # [0=supcon, 1=dropcon, 7=logsum]
    noise_levels = (0, 10, 20, 30, 40, 50)

    for noise in noise_levels:
        run_pretraining(noise_percentage=noise, temperature=temperature, drop_k=drop_k, backbone=backbone, loss=loss)
        run_downstream_finetuning_set(
            noise_percentage=noise,
            epoch_weights=0,
            temperature=temperature,
            drop_k=drop_k,
            backbone=backbone,
            loss=loss,
        )

    for noise in noise_levels:
        run_downstream_set_random_init(noise_percentage=noise, epoch_weights=0, backbone=backbone)

    for noise in noise_levels:
        run_downstream_set_imagenet(noise_percentage=noise, epoch_weights=0, backbone=backbone)


if __name__ == "__main__":
    main()
