import torch
from torch.autograd import Variable
import numpy as np

from network import Network

import os
import io
from PIL import Image

SIZE = 512
NGF = 128
MODEL_CONF = 'StyleTransfer.model'


def load_model():
    model = Network(ngf=NGF)

    model_dict = torch.load(MODEL_CONF)
    model_dict_clone = model_dict.copy()

    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    model.load_state_dict(model_dict, False)

    return model


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)),
                         Image.ANTIALIAS)
    img = np.array(img).transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, SIZE).numpy()
    else:
        img = tensor.clone().clamp(0, SIZE).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def tensor_get_rgbimage(tensor, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, SIZE).numpy()
    else:
        img = tensor.clone().clamp(0, SIZE).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')

    return img_byte_arr.getvalue()


def tensor_get_bgrimage(tensor, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    return tensor_get_rgbimage(tensor, cuda)


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


def clear_image(path):
    if path:
        os.remove(path)


def get_transferred_image(model: Network, style_path, target_path):
    style_image = tensor_load_rgbimage(style_path, size=SIZE).unsqueeze(0)
    style = preprocess_batch(style_image)

    target_image = tensor_load_rgbimage(target_path, size=SIZE,
                                        keep_asp=True).unsqueeze(0)
    target = preprocess_batch(target_image)

    style_v = Variable(style)
    target_v = Variable(target)

    model.set_style(style_v)
    output = model(target_v)

    return tensor_get_bgrimage(output.data[0], False)
