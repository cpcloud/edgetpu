import json
import subprocess
import tempfile
from pathlib import Path

import click


def tflite_to_json(flatbuffer_model_path, json_output_dir, fbs_schema):
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


def json_to_tflite(json_model_path, output_dir, fbs_schema):
    subprocess.check_output(
        ["flatc", "--binary", "-o", output_dir, fbs_schema, json_model_path]
    )


def delete_decoder_from_json(path):
    with path.open("r") as f:
        model = json.load(f)
    del model["operator_codes"][1]
    del model["subgraphs"][0]["operators"][1]

    model["subgraphs"][0]["tensors"] = model["subgraphs"][0]["tensors"][:4]
    model["buffers"] = model["buffers"][:4]
    model["subgraphs"][0]["outputs"] = [0, 1, 2]
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
