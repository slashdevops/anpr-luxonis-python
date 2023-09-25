import argparse

import blobconverter

parser = argparse.ArgumentParser()
parser.add_argument("-bin", "--ov-bin-file", type=argparse.FileType("r"), required=True, help="OpenVINO’s IR format bin")
parser.add_argument("-xml", "--ov-xml-file", type=argparse.FileType("r", encoding="UTF-8"), required=True, help="OpenVINO’s IR format xml")
parser.add_argument("-onn", "--onnx-file", type=argparse.FileType("r", encoding="UTF-8"), help="OpenVINO’s IR format onnx")

parser.add_argument("-o", "--output-dir", type=str, default="models/DepthAI", help="Output directory")

parser.add_argument("-dt", "--data-type", type=str, default="FP16", help="Model data type")
parser.add_argument("-s", "--shaves", type=int, default=6, help="Model data type")
args = parser.parse_args()

# ov parameters and onnx parameters are mutually exclusive
if args.ov_bin_file and args.onnx_file:
    raise RuntimeError("OpenVINO’s IR format bin and OpenVINO’s IR format onnx are mutually exclusive")

if args.ov_bin_file and args.ov_xml_file:
    blob_path = blobconverter.from_openvino(
        xml=args.ov_xml_file.name,
        bin=args.ov_bin_file.name,
        data_type=args.data_type,
        shaves=args.shaves,
        output_dir=args.output_dir,
    )
    print(f"Blob path: {blob_path}")

if args.onnx_file:
    blob_path = blobconverter.from_onnx(
        model=args.onnx_file.name,
        data_type=args.data_type,
        shaves=args.shaves,
        output_dir=args.output_dir,
    )
    print(f"Blob path: {blob_path}")


# python3 converter.py -bin models/DepthAI/vehicle-detection-0200/vehicle-detection-0200.bin -xml models/DepthAI/vehicle-detection-0200/vehicle-detection-0200.xml -o models/DepthAI/vehicle-detection-0200
