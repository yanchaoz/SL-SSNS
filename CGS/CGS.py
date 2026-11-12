"""Command-line workflow for coverage-based EM subvolume selection."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--cfg", default="SL", help="Config basename or YAML path"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "record",
        help="New output directory; existing paths are refused",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=100,
        help="Read the first N z-slices (default: 100); 0 reads the full volume",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip UMAP and plotting"
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots interactively"
    )
    args = parser.parse_args(argv)
    if args.max_slices < 0:
        parser.error("--max-slices must be nonnegative")
    if args.no_plots and args.show_plots:
        parser.error("--no-plots and --show-plots are mutually exclusive")
    if args.output_dir.exists():
        parser.error(f"Output already exists: {args.output_dir}")

    import torch
    from CGS_tools import (
        ProviderValid,
        extract_subvolume,
        load_config,
        load_model,
        run_inference,
        save_results,
    )

    device = torch.device(
        ("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    cfg = load_config(args.cfg)
    provider = ProviderValid(cfg, max_slices=args.max_slices)
    model = load_model(cfg, cfg.MODEL.model_pth, device)
    loader = torch.utils.data.DataLoader(
        provider, batch_size=1, shuffle=False, num_workers=0
    )
    features, positions = run_inference(model, loader, provider, device)
    save_results(
        cfg.CGS.n_neighbors_list,
        cfg.CGS.subvolume_num,
        features,
        positions,
        cfg.CGS.window_size,
        [provider.num_z, provider.num_y, provider.num_x],
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        show_plots=args.show_plots,
    )
    extract_subvolume(
        cfg,
        cfg.CGS.window_size,
        cfg.MODEL.stride,
        cfg.MODEL.crop_size,
        provider.datasets[0],
        output_dir=args.output_dir,
    )
    print(f"Selection saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
