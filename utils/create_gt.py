from os import listdir
from os.path import isfile, join


def create_gt():
    dataset = 'home/cic/datasets/ImageNet/'
    save_folder = '/home/nsallent/alexnet_natural_images/dataset/'

    for folder in listdir(dataset + 'train/'):
        images = [f for f in listdir(folder) if isfile(join(folder, f))]
        with open(save_folder + 'train_gt.txt', 'w') as f:
            im_class = images[0].split('_')
            for _ in range(len(images)):
                f.write("%s\n" % im_class[0])

    for folder in listdir(dataset + 'test/'):
        images = [f for f in listdir(folder) if isfile(join(folder, f))]
        with open(save_folder + 'test_gt.txt', 'w') as f:
            im_class = images[0].split('_')
            for _ in range(len(images)):
                f.write("%s\n" % im_class[0])
