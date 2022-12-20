A NSFW checker for [Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Replaces non-worksafe images with black squares. Install it from UI.

I want to use rickrolling function, so this fork is for testing Rickrolling.

![sample01](https://raw.githubusercontent.com/alansmithee-johndoe/stable-diffusion-webui-nsfw-censor-rickrolling/master/sample01.png)

GPU Cooling interval function is also tested.

![sample02](https://raw.githubusercontent.com/alansmithee-johndoe/stable-diffusion-webui-nsfw-censor-rickrolling/master/sample02.png)

**INSTALL**

In GitHub

![sample03](https://raw.githubusercontent.com/alansmithee-johndoe/stable-diffusion-webui-nsfw-censor-rickrolling/master/sample03.png)

In Web UI's Extensions Tab

![sample04](https://raw.githubusercontent.com/alansmithee-johndoe/stable-diffusion-webui-nsfw-censor-rickrolling/master/sample04.png)

**source of warning-images**

rick.jpeg
https://github.com/CompVis/stable-diffusion/blob/main/assets/rick.jpeg

neko.png
http://scp-jp.wikidot.com/local--files/scp-040-jp/neko.png

**references**

rickrolling
https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py

```
def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x
```

**pass-through**

For example, in scripts/censor.py

```
    return x_checked_image, has_nsfw_concept
```

to

```
    return x_image, has_nsfw_concept
```

**GPU Cooling settings**

Change the value of variable "gpu_cooling_interval = 500" in scripts/censor.py

For example,

```
gpu_cooling_interval = 500
```

to

```
gpu_cooling_interval = 250
```
