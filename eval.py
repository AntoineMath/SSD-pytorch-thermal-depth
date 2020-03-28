import os
import json
from tqdm import tqdm
from pprint import PrettyPrinter
import argparse
import torch
from utils import calculate_mAP
from datasets import ThermalDataset


parser = argparse.ArgumentParser()
parser.add_argument("test_folder", type=str, help="path to the folder containing the .json datafiles")
parser.add_argument("img_type", help="type of image to train on, either 'thermal' or 'depth'", type=str)
parser.add_argument("weights", type=str, help="path to the weights.pth.tar file")
parser.add_argument("train_data", type=str, help="path to the train_data folder containing .json files used for training the model")
parser.add_argument('--min_score', type=float, default=0.01, help="minimum score to consider a detection")
parser.add_argument("--max_overlap", type=float, default=0.45, help="limit of overlapping beyond which we consider there is only one object")
parser.add_argument("-k", "--top_k", type=int, default=1, help="top k possible detections you want the model makes")
parser.add_argument("-r", "--render", action="store_true", help='activate the render of Precision-Recall curves')
args = parser.parse_args()
assert args.img_type in {'thermal', 'depth'}

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters

# training_set : training set used for the training of the weights you want evaluate.
training_set = ThermalDataset(args.train_data, img_type=args.img_type, split='train')
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = args.test_folder
mean, std = training_set.dataset_mean, training_set.dataset_std
checkpoint = args.weights

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = ThermalDataset(args.test_folder,
                              img_type=args.img_type,
                              split='test',
                              mean_std=[mean, std],
                              keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          collate_fn=test_dataset.collate_fn,
                                          num_workers=workers,
                                          pin_memory=True)


def evaluate(test_loader, model, min_score, max_overlap, top_k, render):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    :param min_score: min_score to consider a detection
    :param max_overlap: maximum overlapping between 2 objects to consider both objects. If it exceeds, only 1 object.
    :param top_k: to keep the top k prediction in the image
    :param render: if True, render the PR curves
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):

            images = images.to(device)  # (N, 1, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs,
                                                                                       predicted_scores,
                                                                                       min_score=min_score,
                                                                                       max_overlap=max_overlap,
                                                                                       top_k=top_k)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        class_precisions, class_recalls, APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes,
                                                                  true_labels, true_difficulties, render=render)

        # Creates a json file summarizing the evaluation
        eval_info = {'img_type': args.img_type,
                     'training_data': args.train_data,
                     'evaluation_data': args.test_folder,
                     'weights': args.weights,
                     'min_score': args.min_score,
                     'max_overlap': args.max_overlap,
                     'top_k': args.top_k}

        if not os.path.isdir('eval'):
            os.makedirs('eval')

        with open(os.path.join('eval', args.weights.split('/')[-1].replace('.pth.tar', '.json')), 'a') as file:
            json.dump({'eval_info': eval_info,
                       'class_precisions': class_precisions,
                       'class_recalls': class_recalls,
                       'APs': APs,
                       'mAP': mAP}, file)

    # Print
    print('Precisions:')
    pp.pprint(class_precisions)
    print('\nRecalls:')
    pp.pprint(class_recalls)

    print('\nAP:')
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model, args.min_score, args.max_overlap, args.top_k, args.render)
