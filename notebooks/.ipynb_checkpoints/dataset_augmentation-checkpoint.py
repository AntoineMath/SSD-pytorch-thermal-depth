import os

dataset = '/home/mathurin/prudence/dataset_aug/'
print(os.listdir(dataset))
if __name__ == '__main__':
    for serie in os.listdir(dataset):
        print(serie)
        print('o')
