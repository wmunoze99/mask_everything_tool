import gradio as gr
import cv2
from gradio_image_prompter import ImagePrompter
from pathlib import Path
import torch
from urllib.request import urlretrieve
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from utils.progress import ProgressBarController
from utils.draw_utilities import show_mask, draw_points
import matplotlib.pyplot as plt
from utils.db import DB
from tinydb import Query
from utils.export import export_to_coco, export_to_yolo

progressBar = ProgressBarController()


def main():
    progressBar.new("Creating DB")
    db = DB()
    db.create_table('tags')
    progressBar.terminate()

    progressBar.new("Preparing file system")
    images_result = Path('last_results')

    if not images_result.exists():
        images_result.mkdir()

    images_data = Path('data')

    if not images_data.exists():
        images_data.mkdir()

    images = Path('data/images')
    masks = Path('data/masks')

    if not images.exists():
        images.mkdir()

    if not masks.exists():
        masks.mkdir()

    progressBar.terminate()

    progressBar.new("Preparing and Downloading Model")
    models = Path('models')
    model_path = models / 'sam_vit_h_4b8939.pth'

    if not models.exists():
        models.mkdir()

    if not model_path.exists():
        urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                    model_path)

    progressBar.terminate()

    progressBar.new("Configuring model")
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = 'vit_h'

    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(model_path))
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    progressBar.terminate()

    def process(prompts):
        progressBar.new("Processing image data")

        image = prompts["image"]  # Image getting
        prompt_points = prompts["points"]  # All points inserted into the image
        squares = np.array([])  # Initialize square array. Only one is available for more is needed uses tensors
        points = []  # Points set over the image
        label = []  # Label images

        for i, _points in enumerate(prompt_points):
            if _points[3] > 0 and _points[4] > 0:  # Check if it is a point or a square
                squares = np.array([int(_points[0]), int(_points[1]), int(_points[3]), int(_points[4])])
            else:
                points.append([int(_points[0]), int(_points[1])])  # Added points to array
                label.append(i)

        if squares.size == 0:
            raise gr.Error("No squares draw in the image can't process")

        # Convert points to numpy arrays
        points = np.array(points)
        label = np.array(label)

        predictor.set_image(image)  # Set predictor array

        # Draw points into image
        rec_image = cv2.rectangle(np.copy(image), (int(squares[0]), int(squares[1])),
                                  (int(squares[2]), int(squares[3])), color=(255, 0, 255), thickness=2)

        rec_image = draw_points(rec_image, points)

        progressBar.terminate()
        progressBar.new("Analyzing image")

        if points.size > 0:
            mask, scores, _ = predictor.predict(box=squares, point_coords=points, point_labels=label)
        else:
            mask, scores, _ = predictor.predict(box=squares)
        progressBar.terminate()

        last_channel = mask.shape[0] - 1

        progressBar.new("Saving results")

        cv2.imwrite(str(images_result / 'image.jpg'), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_as_image = np.copy(mask[last_channel, :, :])
        mask_as_image_h, mask_as_image_w = mask_as_image.shape[-2:]
        mask_as_image = mask_as_image.reshape(mask_as_image_h, mask_as_image_w, 1).astype(np.uint8) * 255
        cv2.imwrite(str(images_result / 'mask.jpg'), mask_as_image)

        fig = plt.figure()
        plt.imshow(rec_image)

        show_mask(mask[last_channel, :, :], plt.gca())
        progressBar.terminate()

        gr.Info("Tip: With a single click can add points to adjust mask")

        return fig

    def annotate_mask(label_result):
        image_last_result = cv2.imread(str(images_result / 'image.jpg'))
        mask_last_result = cv2.imread(str(images_result / 'mask.jpg'))

        category_query = Query()
        category = db.query_document('tags', category_query.name == label_result)[0]

        return {
            "image_shape": image_last_result.shape,
            "mask_shape": mask_last_result.shape,
            "label": label_result,
            "coco_export": export_to_coco(mask_last_result, category.doc_id),
            "yolo_export": export_to_yolo(mask_last_result, category.doc_id)
        }

    def get_current_tags():
        meta_data = db.get_all_documents_from_table('metadata')[0]
        __tags = db.get_all_documents_from_table('tags')
        available_tags = {}

        for _tag in __tags:
            try:
                available_tags[_tag['name']] = _tag['references'] / meta_data['total_images'] * 100
            except ZeroDivisionError:
                available_tags[_tag['name']] = 0

        return available_tags

    def create_new_tag(tag):
        if not tag:
            gr.Info('Insert a tag name')
            return get_current_tags(), gr.update(choices=[tag['name'] for tag in db.get_all_documents_from_table('tags')])

        query_tag = Query()

        searching_result = db.query_document('tags', query_tag.name == tag)

        if len(searching_result) > 0:
            raise gr.Error("Tag already exits")

        db.insert_into_table('tags', {
            "name": tag,
            "references": 0
        })

        return get_current_tags(), gr.update(choices=[tag['name'] for tag in db.get_all_documents_from_table('tags')])

    # Interface
    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row():
                with gr.Column(variant='panel'):
                    image_prompt = ImagePrompter(show_label=True, sources=["upload"], label="Insert Image")
                    process_button = gr.Button('Process Mask')
                with gr.Column(variant='panel'):
                    mask_result = gr.Plot(label="Result")

            process_button.click(fn=process, inputs=[image_prompt], outputs=[mask_result])

        with gr.Group():
            with gr.Row():
                with gr.Column():
                    with gr.Row(variant='panel'):
                        tag_input = gr.Textbox(label='New Tag', lines=1)
                        tag_create = gr.Button(value='Create tag')
                    with gr.Row(variant='panel'):
                        labels = gr.Label(value=get_current_tags(), label="Tags")

                with gr.Column():
                    with gr.Row(variant='panel'):
                        tags = [tag['name'] for tag in db.get_all_documents_from_table('tags')]
                        label_choice = gr.Dropdown(tags, label="Available tags")
                        label_button = gr.Button('Preview annotation mask')

                    with gr.Row():
                        with gr.Column(variant='panel'):
                            json_result = gr.JSON(label="Annotations to be added", elem_classes=['json-container'])
                            gr.Button('Add annotation')

                tag_create.click(fn=create_new_tag, inputs=[tag_input], outputs=[labels, label_choice])
                label_button.click(fn=annotate_mask, inputs=[label_choice], outputs=[json_result])

                demo.css = """
                    .json-container {
                        max-height: 300px; /* Adjust the height as needed */
                        .json-holder {
                            height: 91%;
                            overflow-y: scroll; /* Enables vertical scrolling */
                        }
                    }
                    """

    demo.launch()


if __name__ == '__main__':
    main()
