import json

from PIL import Image
from flask import Flask, request, make_response, send_file
from prometheus_flask_exporter import PrometheusMetrics

from documentHandler import create_document
from layerHandler import *
from poreUnmeltedHandler import *
from scaleHandler import *
from visualisator import visualize

URL = 'unstoppablegreen.redirectme.net'
APP_HOST = '192.168.0.103'
APP_PORT = '8080'
app = Flask(__name__)

metrics = PrometheusMetrics(app)

endpoints = ("serve-sample", "get-report")

layerModel = init_layer_model()
poreUnmeltedModel = init_pore_unmelted_model()

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/serve-sample", methods=["GET", "POST"])

def serveSample():
    quality_criteria = {}
    response = make_response()
    if request.method == "POST":
        if "file" not in request.files:
            response.status_code = 204
            return response
        file = request.files["file"]
        if file.filename == "":
            response.status_code = 205
            return response
        if file and allowed_file(file.filename):
            if file.filename != "":
                pil_image = Image.open(file).convert("RGB")
                image = np.array(pil_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                try:
                    scale_length, scale_value = find_scale_length_and_value(image)
                    print(scale_value, scale_length)
                except:
                    response.status_code = 421
                    return response
                if scale_length == -1 or scale_value == -1:
                    response.status_code = 421
                    return response

                layer_result = detect_layer(image, layerModel)
                (
                    x_top,
                    y_top,
                    x_low,
                    y_low,
                    top_approx,
                    low_approx,
                    max_distance,
                    median_distance,
                    min_distance,
                    layer_square,
                ) = process_layer_result(layer_result)

                quality_criteria["max-coat-thickness"] = round(
                    calculate_coat_thickness(image, max_distance, scale_length, scale_value), 2
                )
                quality_criteria["median-coat-thickness"] = round(
                    calculate_coat_thickness(image, median_distance, scale_length, scale_value), 2
                )
                quality_criteria["min-coat-thickness"] = round(
                    calculate_coat_thickness(image, min_distance, scale_length, scale_value), 2
                )

                pore_unmelted_result = detect_pore_unmelted(image, poreUnmeltedModel)

                (
                    pores,
                    unmelts,
                    pore_depth_correlation,
                    avg_pore_square,
                    unmelted_depth_correlation,
                    avg_unmelt_square,
                    quantiles_squares,
                    pores_percentage,
                    unmelts_percentage,
                ) = process_pore_unmelted_result(
                    pore_unmelted_result,
                    calculate_pixels_per_square_micron(
                        scale_length, scale_value, image.shape[0], image.shape[1]
                    ),
                    layer_square,
                    top_approx,
                    low_approx,
                )

                quality_criteria["pore-depth-correlation"] = pore_depth_correlation
                quality_criteria["avg-pore-square"] = avg_pore_square
                quality_criteria["unmelted-depth-correlation"] = unmelted_depth_correlation
                quality_criteria["avg-unmelt-square"] = avg_unmelt_square
                quality_criteria["pores-percentage"] = pores_percentage
                quality_criteria["unmelts-percentage"] = unmelts_percentage
                print(quality_criteria)
                if (request.args.get("marked-image") == 'true' or request.args.get("document-report") == 'true'):
                    base64_img, marked_img = visualize(
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy(),
                        x_top,
                        y_top,
                        x_low,
                        y_low,
                        top_approx,
                        low_approx,
                        pores,
                        unmelts,
                    )
                if (request.args.get("marked-image") == 'true'):
                    quality_criteria["marked-image"] = marked_img
                # cv2.imwrite('F:/marked_image.png', marked_image)
                # document = BytesIO()
                if (request.args.get("document-report") == 'true'):
                    filename = create_document(quality_criteria, base64_img)
                    quality_criteria["report-url"] = 'http://'+URL+':'+APP_PORT+'/get-report?report=' + filename

                response = make_response(json.dumps(quality_criteria))
                response.status_code = 200

    else:
        response.status_code = 400
    return response


@app.route("/get-report", methods=["GET", "POST"])
def getReport():
    return send_file(app.root_path + '/reports/' + request.args.get("report"), as_attachment=True,
                     download_name='report.docx')


app.run(APP_HOST, port=APP_PORT, debug=False)
