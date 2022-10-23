import argparse

import blobconverter

parser = argparse.ArgumentParser()
parser.add_argument("-bin", "--ov-bin-file", type=argparse.FileType("r"), required=True, help="OpenVINO’s IR format bin")
parser.add_argument("-xml", "--ov-xml-file", type=argparse.FileType("r", encoding="UTF-8"), required=True, help="OpenVINO’s IR format xml")
parser.add_argument("-dt", "--data-type", type=str, default="FP16", help="Model data type")
parser.add_argument("-s", "--shaves", type=int, default=6, help="Model data type")
args = parser.parse_args()

blob_path = blobconverter.from_openvino(
    xml=args.ov_xml_file.name,
    bin=args.ov_bin_file.name,
    data_type=args.data_type,
    shaves=args.shaves,
)
