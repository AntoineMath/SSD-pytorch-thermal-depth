from utils import create_data_lists

if __name__ == '__main__':
#    create_data_lists_from_voc(voc07_path='/Users/mathurin/ml/PRuDENCE/SSD/VOC2007',
#                      voc12_path='/Users/mathurin/ml/PRuDENCE/SSD/VOC2012',
#                      output_folder='./')
    create_data_lists('/home/mathurin/prudence/13_01_2020_test/', '.', val_ratio=1)
