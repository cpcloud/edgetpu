import logging

import coloredlogs
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils import dataset_utils
from PIL import Image

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')


def main(args):
    """The main function."""

    labels = dataset_utils.read_label_file(args.label)
    logger.info("Starting engine...")
    engine = ClassificationEngine(args.model)
    logger.info("Classification engine initialized")
    with Image.open(args.image) as img:
        for key, value in engine.classify_with_image(img, top_k=3):
            logger.info("---------------------------")
            logger.info(labels[key])
            logger.info(f"Score: {value}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-l", "--label", required=True)
    p.add_argument("-i", "--image", required=True)

    main(p.parse_args())
