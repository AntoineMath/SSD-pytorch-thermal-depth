from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import skimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = './ckpt/BEST_checkpoint_ssd300_unbalanced.pth.tar'
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
model = checkpoint['model']
model = model.to(device)
model.eval()


def detect(img_array_path, img_path, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # preprocess
    img_array = np.load(img_array_path)
    image = thermal_depth_image_preprocessing(img_array)
    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    print(predicted_locs, predicted_scores)

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
       [img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0]]).unsqueeze(0)
    det_boxes = det_boxes * 300

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to(device).tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return img_array

    # Annotate
    #annotated_image = (original_image-np.amin(original_image)) / (np.amax(original_image)-np.amin(original_image))
    #annotated_image = skimage.transform.resize(annotated_image, (300, 300))
    #annotated_image = annotated_image * 255
    #annotated_image = Image.fromarray(annotated_image)
    annotated_image = np.array(Image.open(img_path))
    annotated_image = (annotated_image - np.amin(annotated_image)) / (np.amax(annotated_image) - np.amin(annotated_image))
    annotated_image = skimage.transform.resize(annotated_image, (300, 300))
    annotated_image = (annotated_image * 255).astype('uint8')
    annotated_image = Image.fromarray(annotated_image)
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
    img_path = '/home/mathurin/prudence/data/Serie_0/Thermique/thermal23.png'
    original_image = Image.open(img_path)
    detect(original_image, min_score=0.20, max_overlap=0.2, top_k=1).show()

    #folder = '/home/mathurin/prudence/fusion/Serie2/'
    #img_list = os.listdir(folder + 'Fusion/')
    #array_list = os.listdir(folder + 'Array/')

    # select random
    #data = list(zip(img_list, array_list))
    #random.shuffle(data)
    #images, annotations = list(zip(*data))
    #img_list = list(images)
    #array_list = list(annotations)

    #for i in range(5):
    #    img_path = folder + 'Fusion/' + img_list[i]
    #    img_array_path = folder + 'Array/' + array_list[i]
    #    result = detect(img_array_path, img_path, min_score=0.20, max_overlap=0.2, top_k=1)
    #    result.show()
