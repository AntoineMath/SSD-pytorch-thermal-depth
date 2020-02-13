import os
import csv
import argparse
import torchvision.transforms.functional as FT
import torch
from utils import rev_label_map, resize
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_scv(folder_path, output_path, weights, min_score=0.2, max_overlap=0.45, top_k=1, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param folder_path: path to the folder where the images are stored
    :param output_path: output .csv file path
    :param weights: path to the weights.pth.tar file
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    assert output_path.endswith(".csv"), "you must specify a .csv output file path"
    assert 0 <= min_score <= 1, "min_score must be type of float and between 0 and 1"
    assert 0 <= max_overlap <= 1, "max_overlap must be type of float and between 0 and 1"
    assert top_k >= 0, "top_k must be between positiv or 0"

    # Load model checkpoint
    checkpoint = weights
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    boxes = list()
    labels = list()
    images_path = list()

    for file in os.listdir(folder_path):
        if not file.endswith(".png"):
            continue
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path)
        img, _ = resize(img, boxes=torch.Tensor([0, 0, 0, 0]), dims=(300, 300))

        # Move to default device
        img = FT.to_tensor(img).type('torch.FloatTensor')
        img = FT.normalize(img, mean=[img.mean()],
                             std=[img.std()])
        img = img.type('torch.FloatTensor').to(device)

        # Forward prop.
        predicted_locs, predicted_scores = model(img.unsqueeze(0))

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                 max_overlap=max_overlap, top_k=top_k)

        # Move detections to the CPU
        det_boxes = det_boxes[0].to('cpu')

        # Decode class integer labels
        det_labels = [rev_label_map[l] for l in det_labels[0].to(device).tolist()]

        # Suppress specific classes, if needed
        for i in range(det_boxes.size(0)):
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue

        images_path.append(img_path)
        boxes.append(det_boxes.tolist())
        labels.append(det_labels)

    # Write the result in a csv
    mode = 'a'
    if os.path.isfile(output_path):
        mode = 'w'

    with open(output_path, mode, newline='') as f:
        fieldnames = ['image_name', 'posture', 'bbox_coords']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(images_path)):
            for j in range(len(labels[i])):
                writer.writerow({'image_name': images_path[i],
                                 'posture': labels[i][j],
                                 'bbox_coords': boxes[i][j]})

    print('\ndone')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", help="path to the folder you want run the detection on")
    parser.add_argument("output_path", help="path where you want to save the .csv file")
    parser.add_argument("weights", type=str, help="path to weights.pth.tar file")
    parser.add_argument("--min_score", type=float, help="minimum score to consider a detection")
    parser.add_argument("--max_overlap", type=float, help="limit of overlaping beyond which we consider there is only one object")
    parser.add_argument("-k", "--top_k", type=int, help="top k possible detections you want the model makes")

    args = parser.parse_args()
    args.__dict__ = {k: v for k, v in args.__dict__.items() if v is not None}

    create_scv(**vars(args))


