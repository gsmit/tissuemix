import albumentations as albu


def flip(image, mask, p=1.0):
    assert len(image.shape) == 3 and image.shape[-1] == 3
    assert len(mask.shape) == 2

    # define augmentations
    transform = albu.Compose([

        # flips and rotations
        albu.VerticalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.RandomRotate90(p=0.5)

    ])

    # apply transformations
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    return transformed_image, transformed_mask


def warp(image, mask, p=1.0):
    assert len(image.shape) == 3 and image.shape[-1] == 3
    assert len(mask.shape) == 2

    # define augmentations
    transform = albu.Compose([

        albu.OneOf([
            # distortions and transformations
            albu.ElasticTransform(alpha=1, sigma=20, alpha_affine=20, border_mode=0, value=0, p=1.0),
            albu.GridDistortion(num_steps=3, distort_limit=0.1, border_mode=0, value=0, p=1.0),
        ], p=p),

        # shift, scale, and rotate
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=360, border_mode=0, value=0, p=1.0),

    ])

    # apply transformations
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    return transformed_image, transformed_mask
