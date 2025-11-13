import math


def get_sigma_schedule(epoch: int, config) -> float:
    """
    Compute sigma (blur) value for current epoch using various scheduling strategies.

    Sigma scheduling controls the blur parameter in Sinkhorn distance:
    - As sigma -> 0: closer to OT_0 (more accurate, more expensive, potentially unstable)
    - As sigma -> infinity: closer to MMD (cheaper, less accurate)

    Args:
        epoch: Current epoch number (0-indexed).
        config: Dataclass config object (DATrainableWeightsSigmaConfig) with schedule parameters.

    Returns:
        Blur value for the current epoch (always >= sigma_min_blur).

    Supported schedules:
        - exponential: sigma = initial * (decay_rate ** epoch)
        - linear: sigma = initial - (initial - final) * (epoch / num_epochs)
        - cosine: sigma = final + 0.5 * (initial - final) * (1 + cos(pi * epoch / num_epochs))
        - step: sigma = initial * (gamma ** (epoch // step_size))
        - polynomial: sigma = (initial - final) * (1 - epoch/num_epochs)^power + final
        - constant: sigma = initial (no decay)

    Note: All schedules enforce sigma >= sigma_min_blur to prevent numerical underflow.
    """
    # Get schedule parameters from dataclass config
    schedule_type = getattr(config, "sigma_schedule_type", "exponential")
    initial_blur = getattr(config, "sigma_initial_blur")
    final_blur = getattr(config, "sigma_final_blur")
    num_epochs = getattr(config, "num_epochs")
    min_blur = getattr(config, "sigma_min_blur", 0)

    if schedule_type == "exponential":
        decay_rate = getattr(config, "sigma_decay_rate")
        sigma = initial_blur * (decay_rate**epoch)

    elif schedule_type == "linear":
        progress = min(epoch / max(num_epochs - 1, 1), 1.0)
        sigma = initial_blur - (initial_blur - final_blur) * progress

    elif schedule_type == "cosine":
        progress = min(epoch / max(num_epochs - 1, 1), 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        sigma = final_blur + (initial_blur - final_blur) * cosine_decay

    elif schedule_type == "step":
        step_size = getattr(config, "sigma_step_size")
        step_gamma = getattr(config, "sigma_step_gamma")
        sigma = initial_blur * (step_gamma ** (epoch // step_size))

    elif schedule_type == "polynomial":
        power = getattr(config, "sigma_poly_power")
        progress = min(epoch / max(num_epochs - 1, 1), 1.0)
        sigma = (initial_blur - final_blur) * ((1 - progress) ** power) + final_blur

    elif schedule_type == "constant":
        sigma = initial_blur

    else:
        raise ValueError(
            f"Unknown sigma schedule type: {schedule_type}. "
            f"Supported: exponential, linear, cosine, step, polynomial, constant"
        )

    # Enforce minimum blur to prevent numerical underflow
    # This is IMPORTANT for long training runs with aggressive decay
    return max(sigma, min_blur)


def get_lambda_grl_schedule(epoch: int, config) -> float:
    """
    Compute lambda_grl (gradient reversal lambda) value for current epoch using scheduling.

    Lambda GRL scheduling gradually increases the gradient reversal strength during training,
    which can help stabilize adversarial training by starting with weaker domain alignment
    and gradually increasing it.

    Args:
        epoch: Current epoch number (0-indexed, including warmup).
        config: Dataclass config object (DAAdversarialConfig) with schedule parameters.

    Returns:
        Lambda GRL value for the current epoch.

    Supported schedules:
        - linear: lambda = start + (end - start) * (epoch / num_epochs)
        - cosine: lambda = end + 0.5 * (start - end) * (1 + cos(pi * epoch / num_epochs))
        - sigmoid: lambda = start + (end - start) * sigmoid((epoch - midpoint) / scale)
        - step: lambda increases in steps
        - polynomial: lambda = (end - start) * (epoch/num_epochs)^power + start
        - constant: lambda = end (no scheduling, use constant value)
    """
    schedule = getattr(config, "lambda_grl_schedule", None)
    if schedule is None:
        # No scheduling, use constant lambda_grl
        return getattr(config, "lambda_grl", 0.25)

    schedule_type = schedule.get("type", "linear").lower()
    start = schedule.get("start", 0.0)
    end = schedule.get("end", 0.25)
    num_epochs = getattr(config, "num_epochs")
    warmup_epochs = getattr(config, "warmup_epochs", 0)

    # Adjust epoch to exclude warmup (scheduling starts after warmup)
    effective_epoch = max(0, epoch - warmup_epochs)
    effective_epochs = num_epochs - warmup_epochs
    if effective_epochs <= 0:
        effective_epochs = 1

    # Progress from 0 to 1 over effective training epochs
    progress = min(effective_epoch / max(effective_epochs - 1, 1), 1.0)

    if schedule_type == "linear":
        lambda_val = start + (end - start) * progress

    elif schedule_type == "cosine":
        # Cosine schedule: starts at start, ends at end
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        lambda_val = start + (end - start) * (1 - cosine_decay)

    elif schedule_type == "sigmoid":
        # Sigmoid schedule: smooth S-curve from start to end
        # Default: midpoint at 50% of training, scale controls steepness
        midpoint = schedule.get("midpoint", 0.5)  # Fraction of training (0.0 to 1.0)
        scale = schedule.get("scale", 0.1)  # Controls steepness (smaller = steeper)
        # Map progress to sigmoid: sigmoid((x - midpoint) / scale)
        x = (progress - midpoint) / scale
        sigmoid_val = 1.0 / (1.0 + math.exp(-x))
        lambda_val = start + (end - start) * sigmoid_val

    elif schedule_type == "step":
        # Step schedule: increases in discrete steps
        step_size = schedule.get("step_size", max(1, effective_epochs // 3))
        num_steps = effective_epoch // step_size
        max_steps = (effective_epochs - 1) // step_size
        if max_steps > 0:
            step_progress = min(num_steps / max_steps, 1.0)
        else:
            step_progress = 1.0
        lambda_val = start + (end - start) * step_progress

    elif schedule_type == "polynomial":
        # Polynomial schedule: lambda = (end - start) * progress^power + start
        power = schedule.get("power", 2.0)
        lambda_val = start + (end - start) * (progress**power)

    elif schedule_type == "constant":
        lambda_val = end

    else:
        raise ValueError(
            f"Unknown lambda_grl schedule type: {schedule_type}. "
            f"Supported: linear, cosine, sigmoid, step, polynomial, constant"
        )

    return float(lambda_val)
