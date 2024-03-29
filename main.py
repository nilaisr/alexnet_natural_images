import argparse
from methods import two_streams_rgb
from utils import create_gt

models_refs = {
    'two_streams_rgb': two_streams_rgb
}


def main():
    parser = argparse.ArgumentParser(description='Classify images.')

    parser.add_argument('-gt', '--create_gt', action='store_true', help='Create gt')
    parser.add_argument('model', help='Model to execute', choices=models_refs.keys())

    args = parser.parse_args()
    create_gt_bool = args.create_gt

    if not create_gt_bool:
        create_gt()

    model = models_refs.get(args.model)

    model()


if __name__ == '__main__':
    main()
