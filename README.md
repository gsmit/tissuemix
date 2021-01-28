# TissueMix
TissueMix is a remarkably simple cutting and pasting augmentation technique designed explicitly for semantic segmentation tasks using digitized histopathology images. Its underlying idea is to synthetically increase the size of small histopathology datasets by re-using existing annotated tissue patches to craft new unseen compositions. This is done by cutting out a target object from image *A*, applying a variety of augmentations, and pasting it into a different background image *B*.

# Usage

To apply TissueMix on two images, simply copy this repo and run the following snippet:

```python
import tissuemix as tm
import matplotlib.pyplot as plt

# load target patches
target = tm.load_image(f'../images/targets/target_03.png')
target_mask = tm.load_mask(f'../images/targets/target_03_mask.png')

# load background patches
background = tm.load_image(f'../images/backgrounds/background_03.png')
background_mask = tm.load_mask(f'../images/backgrounds/background_03_mask.png')

# apply tissuemix
x, y = tm.apply_tissuemix(
    target=target,
    target_mask=target_mask,
    background=background,
    background_mask=background_mask,
    blend=True,
    color=True,
    warp=True)
    
# show mixed target and mask
tm.show_image(x, y)
```
