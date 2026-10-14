"""Research adapter: 'Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection' (MIT, HF yermandy/deepfake-detection).
CLIP ViT-L/14 (LN-tuned), TorchScript bf16. Preprocessing = CLIPProcessor(openai/clip-vit-large-patch14):
resize shortest side 224 (bicubic) -> center crop 224 -> CLIP mean/std. Output logits [real, fake]."""
import numpy as np
import torch
from PIL import Image

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def clip_preprocess(rgb):
    im = Image.fromarray(rgb)
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.Resampling.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    a = (np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255 - MEAN) / STD
    return a.transpose(2, 0, 1)


class YermandyRunner:
    def __init__(self, path, threads=2, dtype=torch.bfloat16):
        torch.set_num_threads(threads)
        self.m = torch.jit.load(path, map_location="cpu").eval()
        self.dtype = dtype
        if dtype != torch.bfloat16:
            self.m = self.m.to(dtype)

    @torch.inference_mode()
    def p_fake(self, crops):
        x = torch.from_numpy(np.stack([clip_preprocess(c) for c in crops])).to(self.dtype)
        lg = self.m(x).float()
        return torch.softmax(lg, 1)[:, 1].numpy(), lg.numpy()
