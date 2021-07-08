import itertools
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import IO

import click
import ujson


def tflite_to_json(data: IO[bytes], json_output_dir: Path, fbs_schema: Path) -> Path:
    with tempfile.NamedTemporaryFile(mode="r+b") as f:
        f.write(data.read())
        f.seek(0)
        subprocess.check_output(
            [
                "flatc",
                "--json",
                "--strict-json",
                "--defaults-json",
                "-o",
                str(json_output_dir),
                str(fbs_schema),
                "--",
                f.name,
            ],
        )
    return next(json_output_dir.glob("*.json"))


def json_to_tflite(json_model_path: str, output_dir: str, fbs_schema: str) -> None:
    subprocess.check_output(
        ["flatc", "--binary", "-o", output_dir, fbs_schema, json_model_path]
    )


def delete_decoder_from_json(path: Path):
    with path.open("r") as f:
        model = ujson.load(f)
    model["operator_codes"] = [
        code
        for code in model["operator_codes"]
        if code.get("custom_code") != "PosenetDecoderOp"
    ]
    model["subgraphs"][0]["operators"] = [
        op for op in model["subgraphs"][0]["operators"] if op["opcode_index"] != 6
    ]

    model["subgraphs"][0]["tensors"] = [
        t for t in model["subgraphs"][0]["tensors"] if "poses" not in t["name"]
    ]

    del model["buffers"][-4:]
    model["subgraphs"][0]["outputs"] = list(
        itertools.chain.from_iterable(
            op["outputs"] for op in model["subgraphs"][0]["operators"][-3:]
        )
    )
    return model


@click.command()
@click.option("-i", "--input", type=click.File("rb"), default=sys.stdin.buffer)
@click.option("-o", "--output", type=click.File("wb"), default=sys.stdout.buffer)
@click.option("-s", "--fbs-schema", type=click.Path(exists=True), required=True)
def main(input: IO[bytes], output: IO[bytes], fbs_schema: str) -> None:
    """Remove the decoding step from a pose edgetpu model."""

    with tempfile.TemporaryDirectory() as json_output_dir:
        s = tflite_to_json(input, Path(json_output_dir), Path(fbs_schema)).read_bytes()
    output.write(s)


if __name__ == "__main__":
    main()
