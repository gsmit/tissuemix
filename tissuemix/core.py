import random
import numpy as np
import tissuemix as tm


def get_stack(m):
    assert len(m.shape) == 2
    return np.stack([m] * 3, axis=-1).astype(int)


def get_mask(s):
    assert len(s.shape) == 3 and s.shape[-1] == 3
    return s[:, :, 0].astype(int)


def get_cutout(i, m):
    assert len(i.shape) == 3
    assert len(m.shape) == 2

    # remove background
    stack = get_stack(m)
    cutout = np.copy(i)
    cutout[(stack == 0)] = 255
    return cutout


def mix(target, target_mask, background, background_mask, transform=None):
    assert len(target_mask.shape) == 2
    assert len(background_mask.shape) == 2

    # convert mask to stack
    target_stack = get_stack(target_mask)

    # cut and paste target
    output = np.copy(background)
    output[(target_stack == 1)] = target[(target_stack == 1)]
    output_mask = np.maximum(target_mask, background_mask)
    return output, output_mask


def apply_tissuemix(target, target_mask, background, background_mask,
                    blend=False, color=False, warp=False):

    target_mask = tm.filter_blobs(np.copy(target_mask))

    if color:
        # apply color correction
        target, background = tm.transfer_color(target, background)

    if warp:
        # apply warping to image
        target, target_mask = tm.warp(image=target, mask=target_mask)
        background, background_mask = tm.flip(image=background, mask=background_mask)

    if blend and random.random() >= 0.5:
        # apply gaussian blur
        x = tm.gaussian_blur(target, target_mask, background)
        y = np.copy(target_mask)
    elif blend and random.random() < 0.5:
        # apply poisson blending
        x = tm.poisson_blend(target, target_mask, background)
        y = np.copy(target_mask)
    else:
        # no blending
        x, y = tm.mix(target, target_mask, background, background_mask)

    return x, y

