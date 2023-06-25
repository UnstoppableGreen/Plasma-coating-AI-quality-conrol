from statistics import *
from statistics import mean

import numpy as np
import scipy
from mmdet.apis import init_detector, inference_detector
from mmdet.structures.mask import bitmap_to_polygon


def init_pore_unmelted_model():
    # Specify the path to model config and checkpoint file
    config_file = "./models/pore-unmelted-inference/config-pore-unmelted.py"
    checkpoint_file = "./models/pore-unmelted-inference/epoch_36.pth"

    # Build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device="cpu")

    return model


def detect_pore_unmelted(img, model):
    result = inference_detector(model, img)
    return result


def process_pore_unmelted_result(
    result, pixels_per_square_micron, layer_square, top_approx, low_approx
):
    pore_polygons = []
    unmelted_polygons = []

    pores = []
    unmelts = []

    sum_squares = []
    sum_squares.append(0)
    sum_squares.append(0)

    for index, mask in enumerate(result.pred_instances.masks):
        poly = bitmap_to_polygon(mask)
        polyArea = PolyArea(poly[0][0])
        sum_squares[int(result.pred_instances.labels[index])] += polyArea
        x = []
        y = []
        x, y = zip(*poly[0][0])
        x_center, y_center = PolyCentroid(poly[0][0])
        if (
            point_position(x_center, y_center, top_approx[0], top_approx[1])
            == "below"
            or "on"
        ):
            if (
                point_position(x_center, y_center, low_approx[0], low_approx[1])
                == "above"
                or "on"
            ):
                if int(result.pred_instances.labels[index]) == 0:
                    pore_polygons.append([polyArea, (x_center, y_center)])
                    pores.append([x, y])
                else:
                    unmelted_polygons.append([polyArea, (x_center, y_center)])
                    unmelts.append([x, y])

    squares = []
    coordinates = []

    # Коэффициент корелляция глубины пор и площади
    squares, coordinates = zip(*pore_polygons)
    pore_depth_correlation = round(
        scipy.stats.pearsonr(squares, list(list(zip(*coordinates))[0]))[0], 2
    )

    # Средняя площадь пор
    avg_pore_square = round(stdev(squares, mean(squares)) / pixels_per_square_micron, 2)

    # Коэффициент корелляция глубины непроплавов и площади
    print(unmelted_polygons)
    if (unmelted_polygons):
        squares, coordinates = zip(*unmelted_polygons)
        unmelted_depth_correlation = round(
            scipy.stats.pearsonr(squares, list(list(zip(*coordinates))[0]))[0], 2
        )
    else:
        unmelted_depth_correlation = 0
    # Средняя площадь непроплавов
    avg_unmelt_square = round(
        stdev(squares, mean(squares)) / pixels_per_square_micron, 2
    )

    # Наиболее часто встречаемые площади пор
    quantiles_squares = quantiles(squares)
    for quantile in quantiles_squares:
        quantile = round(quantile / pixels_per_square_micron, 2)

    # Процент пористости
    pores_percentage = round(100 * float(sum_squares[0]) / float(layer_square), 2)
    # Процент непроплавов
    unmelts_percentage = round(100 * float(sum_squares[1]) / float(layer_square), 2)

    return (
        pores,
        unmelts,
        pore_depth_correlation,
        avg_pore_square,
        unmelted_depth_correlation,
        avg_unmelt_square,
        quantiles_squares,
        pores_percentage,
        unmelts_percentage,
    )


def PolyCentroid(vertices):
    x, y = 0, 0
    n = len(vertices)
    signed_area = 0
    for i in range(len(vertices)):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        # shoelace formula
        area = (x0 * y1) - (x1 * y0)
        signed_area += area
        x += (x0 + x1) * area
        y += (y0 + y1) * area
    signed_area *= 0.5
    x /= 6 * signed_area
    y /= 6 * signed_area
    if isNaN(x):
        x = mean(list(zip(*vertices))[0])
    if isNaN(y):
        y = mean(list(zip(*vertices))[1])
    return int(x), int(y)


def isNaN(num):
    return num != num


def PolyArea(poly):
    x = []
    y = []
    x, y = zip(*poly)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def point_position(x, y, k, b):
    """
    Функция для проверки положения точки относительно прямой y = kx + b
    :param x: координата X точки
    :param y: координата Y точки
    :param k: угловой коэффициент прямой
    :param b: коэффициент смещения от оси Y
    :return: 'above', если точка находится над прямой, 'below', если под прямой, 'on', если на прямой
    """
    y_on_line = k * x + b
    if y > y_on_line:
        print ("above")
        return "above"
    elif y < y_on_line:
        print("below")
        return "below"
    else:
        print("on")
        return "on"
