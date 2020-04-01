import numbers

import numpy as np
import torch


class MultiCrop(object):
    """
    Crop the given PIL Image into multiple random crops

    Parameters
    ----------
    size : tuple or int
        Desired output size of the crop. If size is an ``int`` instead of
        sequence like (h, w), a square crop of size (size, size) is made.

    n_crops : int
        The number of crops to be generated from a page.

    Returns
    -------
    None

    Example
    -------
        >>> MultiCrop(size=model_expected_input_size, n_crops=multi_crop),
        >>> transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        >>> transforms.Lambda(lambda items: torch.stack([transforms.Normalize(mean=mean, std=std)(item) for item in items]))

        >>> transform = Compose([
        >>> MultiCrop(size), # this is a list of PIL Images
        >>> Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
        >>> ])
        >>> #In your test loop you can do the following:
        >>> input, target = batch # input is a 5d tensor, target is 2d
        >>> bs, ncrops, c, h, w = input.size()
        >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
        >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

    """
    def __init__(self, size, n_crops):
        # TODO: DOES NOT PLAY WELL WITH SEEDS. Figure out why!
        self.size = size
        self.n_crops = n_crops
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return multi_crop(img, self.size, self.n_crops)


def multi_crop(img, size, n_crops):
    """
    Crop the given PIL Image into multiple random crops.

    Parameters
    ----------
    img : PIL.Image
        The Image to be processed.
    size : tuple or int
        Desired output size of the crop. If size is an ``int`` instead of
        sequence like (h, w), a square crop of size (size, size) is made.
    n_crops : int
        The number of crops to be generated from a page.

    Returns
    -------
    crops : list of PIL.Images
        A list of PIL.Images which are the crops from the page.

    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))

    crops = []
    for i in range(n_crops):
        x = np.random.randint(0, w - crop_w)
        y = np.random.randint(0, h - crop_h)
        assert x + crop_w < w
        assert y + crop_h < h
        crops.append(img.crop((x, y, x + crop_w, y + crop_h)))
    return crops
