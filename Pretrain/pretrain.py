"""Self-supervised encoder pretraining with explicit configuration and device handling."""

import argparse
import logging
from datetime import datetime
from pathlib import Path


def build_model(cfg, device):
    import torch
    from model.encoder import UNet_PNI_encoder

    model = UNet_PNI_encoder(
        in_planes=cfg.MODEL.input_nc,
        filters=cfg.MODEL.filters,
        pad_mode=cfg.MODEL.pad_mode,
        bn_mode=cfg.MODEL.bn_mode,
        relu_mode=cfg.MODEL.relu_mode,
        init_mode=cfg.MODEL.init_mode,
    ).to(device)
    if cfg.MODEL.pre_train:
        checkpoint = torch.load(
            cfg.MODEL.pretrain_path, map_location=device, weights_only=True
        )
        state = checkpoint.get("model_weights", checkpoint)
        model.load_state_dict(
            {key.removeprefix("module."): value for key, value in state.items()}
        )
    count = (
        torch.cuda.device_count()
        if device.type == "cuda" and device.index in (None, 0)
        else 0
    )
    if count > 1:
        if cfg.TRAIN.batch_size % count:
            raise ValueError("Batch size must divide evenly across visible GPUs")
        model = torch.nn.DataParallel(model)
    return model


def loop(cfg, provider, learner, optimizer, iteration, writer):
    import torch
    from training_utils import calculate_lr, save_checkpoint

    total_loss, steps = 0.0, 0
    with (Path(cfg.record_path) / "loss.txt").open("a", encoding="utf-8") as record:
        while iteration < cfg.TRAIN.total_iters:
            iteration += 1
            learner.train()
            first, second = provider.next()
            lr = calculate_lr(cfg, iteration)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad()
            loss = learner(first, second)
            loss.backward()
            if cfg.TRAIN.weight_decay is not None:
                with torch.no_grad():
                    for group in optimizer.param_groups:
                        for parameter in group["params"]:
                            parameter.mul_(1 - cfg.TRAIN.weight_decay * group["lr"])
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if iteration == 1 or iteration % cfg.TRAIN.display_freq == 0:
                mean_loss = total_loss / steps
                logging.info(
                    "step %d, loss = %.6f, lr = %.8f", iteration, mean_loss, lr
                )
                writer.add_scalar("loss", mean_loss, iteration)
                record.write(f"step = {iteration}, loss = {mean_loss}\n")
                record.flush()
                total_loss, steps = 0.0, 0
            if (
                iteration % cfg.TRAIN.save_freq == 0
                or iteration == cfg.TRAIN.total_iters
            ):
                save_checkpoint(
                    Path(cfg.save_path) / f"model-{iteration:06d}.ckpt",
                    learner,
                    optimizer,
                    iteration,
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--cfg",
        default="pretraining_snemi3d",
        help="Config basename or YAML path",
    )
    parser.add_argument("-m", "--mode", choices=["train"], default="train")
    parser.add_argument("--device", help="cpu or cuda:N; otherwise use TRAIN.if_cuda")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Full training checkpoint to resume into a new run directory",
    )
    args = parser.parse_args(argv)

    import torch
    from learner import BSL
    from pretrain_provider import Provider
    from tensorboardX import SummaryWriter
    from training_utils import load_config, restore_checkpoint
    from utils.utils import setup_seed

    base = Path(__file__).resolve().parent
    path = Path(args.cfg)
    if path.suffix not in (".yaml", ".yml"):
        path = base / "config" / f"{args.cfg}.yaml"
    cfg = load_config(path)
    if cfg.TRAIN.resume and args.resume is None:
        parser.error("Use --resume PATH to select a full training checkpoint")
    device = torch.device(args.device or ("cuda:0" if cfg.TRAIN.if_cuda else "cpu"))
    if device.type not in ("cpu", "cuda"):
        parser.error("Supported devices are cpu and cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; install a compatible build or use --device cpu"
        )
    cfg.TRAIN.if_cuda, cfg.TRAIN.device = device.type == "cuda", str(device)
    if cfg.TRAIN.batch_size < 2:
        parser.error("Self-supervised pretraining requires batch_size >= 2")
    for field in ("total_iters", "save_freq", "display_freq"):
        if getattr(cfg.TRAIN, field) <= 0:
            parser.error(f"TRAIN.{field} must be positive")
    for section, field in (
        (cfg.DATA, "data_folder"),
        (cfg.MODEL, "pretrain_path"),
        (cfg.TRAIN, "save_path"),
        (cfg.TRAIN, "cache_path"),
    ):
        value = Path(getattr(section, field))
        setattr(section, field, str(value if value.is_absolute() else base / value))
    setup_seed(cfg.TRAIN.random_seed)
    provider = Provider("train", cfg)
    model = build_model(cfg, device)
    learner = BSL(
        net=model,
        hidden_layer=-1,
        use_momentum=False,
        projection_size=128,
        projection_hidden_size=512,
    )
    optimizer = torch.optim.Adam(
        learner.parameters(),
        lr=cfg.TRAIN.base_lr,
        betas=(0.9, 0.999),
        eps=0.01,
        weight_decay=1e-6,
        amsgrad=True,
    )
    iteration = (
        restore_checkpoint(args.resume, learner, optimizer, device)
        if args.resume
        else 0
    )
    if iteration >= cfg.TRAIN.total_iters:
        parser.error("Checkpoint has already reached TRAIN.total_iters")
    name = datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f") + "_" + cfg.NAME
    cfg.save_path = str(Path(cfg.TRAIN.save_path) / name)
    cfg.record_path = str(Path(cfg.save_path) / "logs")
    Path(cfg.record_path).mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(Path(cfg.record_path) / "train.log"),
            logging.StreamHandler(),
        ],
    )
    with SummaryWriter(cfg.record_path) as writer:
        writer.add_text("cfg", str(cfg))
        loop(cfg, provider, learner, optimizer, iteration, writer)


if __name__ == "__main__":
    main()
