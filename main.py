import cv2
import numpy as np
import os
import argparse
from dotenv import load_dotenv
from PIL import Image
import io
import base64
import openai
import tqdm

import vtii

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("prompt_mode")
    args = parser.parse_args()
    vtii.video_to_instruction(args.filepath, args.prompt_mode)
