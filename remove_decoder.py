import json
import subprocess
import tempfile
import itertools
from pathlib import Path

import click


def tflite_to_json(
    flatbuffer_model_path: str, json_output_dir: str, fbs_schema: str
) -> None:
    subprocess.check_output(
        [
            "flatc",
            "--json",
            "--strict-json",
            "--defaults-json",
            "-o",
            json_output_dir,
            fbs_schema,
            "--",
            flatbuffer_model_path,
        ],
    )


def json_to_tflite(json_model_path: str, output_dir: str, fbs_schema: str) -> None:
    subprocess.check_output(
        ["flatc", "--binary", "-o", output_dir, fbs_schema, json_model_path]
    )


def delete_decoder_from_json(path: Path):
    with path.open("r") as f:
        model = json.load(f)
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
@click.option("-i", "--input", type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", type=click.Path(), required=True)
@click.option("-s", "--fbs-schema", type=click.Path(exists=True), required=True)
def main(input, output_dir, fbs_schema):
    """Remove the decoding step from a pose edgetpu model."""

    with tempfile.TemporaryDirectory() as json_output_dir:
        print("converting tflite to JSON")
        tflite_to_json(input, json_output_dir, fbs_schema)

        model_path = Path(input)
        model_basename = model_path.with_suffix(".json").name
        output_model_basename = f"{model_path.with_suffix('').name}_no_decoder.tflite"
        input_json_path = Path(json_output_dir) / model_basename

        print("removing decoder")
        model = delete_decoder_from_json(input_json_path)

        output_json_path = Path(json_output_dir) / output_model_basename
        with output_json_path.open("w") as f:
            json.dump(model, f)

        print("converting JSON to tflite")
        json_to_tflite(str(output_json_path), output_dir, fbs_schema)


if __name__ == "__main__":
    main()
