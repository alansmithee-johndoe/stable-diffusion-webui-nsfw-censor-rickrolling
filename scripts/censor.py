# coding: utf-8

import torch

import numpy as np
from tqdm import tqdm
from time import sleep
from os import listdir
from random import choice

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image

from modules import scripts, shared

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None

# for Rickroll
safety_images_dir = "extensions/stable-diffusion-webui-nsfw-censor/warning-images/"
safety_images = listdir(safety_images_dir)
rickrolling_message = "🤗HAHAHA🤗" + "\n\n" + "Rickrolled? Calm down, brother. Listen to his music!" + "\n\n"

# GPU cooling interval
gpu_cooling_interval = 500


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# check and replace nsfw content
def check_safety(x_image):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    return x_checked_image, has_nsfw_concept


def censor_batch(x):
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim_numpy)
    x = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

    # GPU cooling
    print("\n\n" + "GPU COOLING TIME STARTED")
    for i in tqdm (range (gpu_cooling_interval), desc="Waiting for cooling..."):
        sleep(0.001)
    print("GPU COOLING TIME ENDED" + "\n\n")

    # Rickrolling
    counter = 0
    for unsafe_value in has_nsfw_concept:
        try:
            if unsafe_value == True:
                print(rickrolling_message)
                hwc = x.shape
                y = Image.open(safety_images_dir + choice(safety_images)).convert("RGB").resize((hwc[3], hwc[2]))
                y = (np.array(y)/255.0).astype("float32")
                y = torch.from_numpy(y)
                y = torch.unsqueeze(y, 0).permute(0, 3, 1, 2)
                assert y.shape == x.shape
                x[counter] = y
            counter += 1
        except Exception:
            print("Potential NSFW content was detected in one or more images.")
            counter += 1
    return x


class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs['images']
        images[:] = censor_batch(images)[:]
