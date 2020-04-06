import os 
import csv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("xml_thermal", type=str, help="path to thermal detection file.csv")
parser.add_argument("xml_depth", type=str, help="path to depth detection file.csv")
parser.add_argument("save_path", type=str, help="path where to save the output file.csv")
parser.add_argument("--gt", type=str, help="path to the folder containing TEST_thermal/depth_images/objects.json")
args = parser.parse_args()

xml_thermal = args.xml_thermal
xml_depth = args.xml_depth
save_path = args.save_path
labels_path = args.gt 

dict_labels = ['lying_down', 'standing', 'sitting', 'fall']

if labels_path is not None:
    obj_json = os.path.join(labels_path, 'TEST_thermal_objects.json')
    img_json = os.path.join(labels_path, 'TEST_thermal_images.json')

    with open(img_json, 'r') as f:
       images_json = [img.split('/')[-1] for img in json.load(f)] 

    with open(obj_json, 'r') as f:
        objects_json = [obj['labels'] for obj in json.load(f)]

with open(xml_thermal) as f:
    csv_reader = csv.reader(f, delimiter=',')
    thermal_img, thermal_postures = list(zip(*csv_reader))[:2]
with open(xml_depth) as f:
    csv_reader = csv.reader(f, delimiter=',')
    depth_img, depth_postures = list(zip(*csv_reader))[:2]

assert len(thermal_img) == len(depth_img)
assert len(thermal_postures) == len(depth_postures)
assert len(thermal_postures) == len(thermal_img)
assert len(depth_postures) == len(depth_img)

# reorganize depth_img
depth_img_indices = [depth_img.index(i.replace('Thermique', 'Profondeur').replace('thermal', 'depth')) for i in thermal_img]
depth_postures = [depth_postures[i] for i in depth_img_indices]

# compare
with open(save_path, 'w') as f:
    fieldnames = ['image', 'detect_thermal', 'detect_depth']
    if labels_path is not None:
        fieldnames.append('gt')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(len(thermal_postures)):
        if thermal_postures[i] != depth_postures[i]:
            row = {'image': thermal_img[i],
                   'detect_thermal': thermal_postures[i],
                   'detect_depth': depth_postures[i]}
            if labels_path is not None:
                row['gt'] = ",".join([dict_labels[i-1] for i in objects_json[images_json.index(thermal_img[i].split('/')[-1])]])
            writer.writerow(row)
            print(row)

print('DONE')

            
