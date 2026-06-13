import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def setup_chinese_font():
    font = {
        "family": "MicroSoft YaHei",
        "weight": "bold",
        "size": "10",
    }
    matplotlib.rc("font", **font)


def imshow(img, title=None):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

