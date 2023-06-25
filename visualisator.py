import base64
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def visualize(img, x_top, y_top, x_low, y_low, top_approx, low_approx, pores, unmelts):

    dpi = 96
    figsize = img.shape[1] / float(dpi), img.shape[0] / float(dpi)

    fig = Figure(figsize=figsize, frameon=False, dpi=dpi)
    ax = fig.subplots()

    ax.set_xlim((0, img.shape[1]))
    ax.set_ylim((img.shape[0], 0))

    ax.imshow(img)

    x_top_median = []
    y_top_median = []
    x_low_median = []
    y_low_median = []

    i = 0
    while i <= img.shape[1]:
        x_top_median.append(i)
        x_low_median.append(i)
        y_top_median.append(top_approx[0] * i + top_approx[1])
        y_low_median.append(low_approx[0] * i + low_approx[1])
        i += 1

    for pore in pores:
        ax.plot(pore[0], pore[1], color="magenta", linewidth=0.2)
        ax.fill(pore[0], pore[1], facecolor="magenta", linewidth=0.1, alpha=0.5)

    for unmelt in unmelts:
        ax.plot(unmelt[0], unmelt[1], color="aqua", linewidth=0.2)
        ax.fill(unmelt[0], unmelt[1], facecolor="aqua", linewidth=0.1, alpha=0.5)

    ax.plot(x_top, y_top, x_low, y_low, color="red")
    ax.plot(x_top_median, y_top_median, color="white", linestyle="dashed")
    ax.plot(x_low_median, y_low_median, color="white", linestyle="dashed")

    fig = compact_fig(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return data, f"<img src='data:image/png;base64,{data}'/>"
    # return data


def compact_fig(fig=None):

    if not fig:
        fig = plt.gcf()
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis("off")
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    return fig
