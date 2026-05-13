from utils import *
import json
import sys
import os


def main(dataset):
    
    try:
        seed = int(sys.argv[2])
    except:
        seed = 2025
    print(f'dataset:{dataset}')

    config_path = os.path.join('../datasets',dataset,'config.json')
    with open(config_path,'r') as f:
        config = json.load(f)

    if seed is not None:
        set_random_seed(seed)
        print(f'seed:{seed}')
    else:
        print('no seed')

    feats, labels = run_train(config, train=True)

    save_vis(labels, feats, config)

    
if __name__ == '__main__':
    dataset = sys.argv[1]
    main(dataset)
    