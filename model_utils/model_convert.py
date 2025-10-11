#!/usr/bin/env python3
"""
model_convert.py

Convert PyTorch models to ONNX (if PyTorch available). Simple CLI wrapper.
"""

import argparse
import os


def convert_pytorch_to_onnx(model_path: str, output_path: str, input_shape=(1,3,224,224)):
    try:
        import torch
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        dummy = torch.randn(*input_shape)
        torch.onnx.export(model, dummy, output_path, opset_version=13)
        print(f"Saved ONNX model to {output_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shape', nargs='+', type=int, help='Input shape e.g. 1 3 224 224')
    args = parser.parse_args()

    shape = tuple(args.shape) if args.shape else (1,3,224,224)
    convert_pytorch_to_onnx(args.model, args.output, shape)


if __name__ == '__main__':
    main()
