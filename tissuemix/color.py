import cv2
from color_transfer import color_transfer


def transfer_color(target, background):
    target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    target = color_transfer(background, target)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    return target, background

