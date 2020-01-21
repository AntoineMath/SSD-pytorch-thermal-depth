from torchvision import transforms
import torchvision.transforms.functional as FT
import torch
from utils import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = './ckpt/BEST_checkpoint_ssd300_fusion.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
#resize = transforms.Resize((300, 300))
#to_tensor = transforms.ToTensor()
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])


def detect(img_path, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    original_image = np.load(img_path)
    image = thermal_depth_image_preprocessing(original_image, split='val')
    image = image.type('torch.FloatTensor').to(device)
    #original_image = Image.open(img_path)
    #image = FT.to_tensor(original_image)



    # Move to default device
    #image, _ = resize(original_image, boxes=torch.Tensor([0, 0, 0, 0]), dims=(300, 300))
    #image = FT.to_tensor(image)
    #image = FT.normalize(image, mean=[image.type('torch.FloatTensor').mean()],
    #                         std=[image.type('torch.FloatTensor').std()])


    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.shape[1], original_image.shape[0], original_image.shape[1], original_image.shape[0]]).unsqueeze(0)
    #det_boxes = det_boxes * original_dims
    det_boxes = det_boxes * 300

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    # TODO : Renommer correctement les images dans le dataset au lieu de faire des replace
    annotated_image = Image.open(img_path.replace('Array', 'Fusion').replace('npy', 'png').replace('thermal_depth', 'Final'))
    annotated_image, _ = resize(annotated_image, boxes=torch.Tensor([0, 0, 0, 0]), dims=(300, 300))
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


if __name__ == '__main__':

    # Test sur la serie 4 (qui n'a pas ete annotee)
    folder = '/home/mathurin/prudence/fusion/Serie4/'
    img_list = os.listdir(folder + 'Array')

    # select random
    random.shuffle(img_list)

    for i in range(10):
        img_path = folder + 'Array/' + img_list[i]
        result = detect(img_path, min_score=0.20, max_overlap=0.2, top_k=1)
        result.show()