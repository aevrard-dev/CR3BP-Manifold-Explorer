import argparse
from pathlib import Path
from src.assembler import Assembler


def parse_args():
    parser = argparse.ArgumentParser(description="CR3BP Manifold Explorer")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        config_file = (Path(__file__).parent / args.config).resolve()
        print(f'\nConfiguration file specified: {config_file}')
        app = Assembler(config_file_path=config_file)
        app.process_projection()
    else:
        print(f'\n\n\n'
              f'Running with file: '
              f'\n{(Path(__file__).parent/"config/page76.yaml").resolve()}')
        app = Assembler(config_file_path=(Path(__file__).parent/"config/page76.yaml").resolve())
        print('here', app.numerical)
        app.process_projection()


if __name__ == "__main__":
    main()
