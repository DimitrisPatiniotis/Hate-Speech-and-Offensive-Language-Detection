import numpy as np

BASE_DIR = '../'
DATA_CSV = BASE_DIR + 'Dataset/labeled_data.csv'

num_to_class = {
    0:'hate speech',
    1:'offensive language',
    2:'neither'
}

def set_seed(num=50):
    np.random.seed(num)

if __name__ == '__main__':
    print('General Settings Util')