import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import Dataset
from metrics import Metrics
from model import Model
import numpy as np


def _eval(path_to_checkpoint: str, path_to_data_dir: str, path_to_results_dir: str):
    os.makedirs(path_to_results_dir, exist_ok=True)

    model = Model()
    model.load(path_to_checkpoint).eval()

    dataset = Dataset(path_to_data_dir, Dataset.Mode.EVAL)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    print('Start evaluating')

    with torch.no_grad():
        merged_logits = torch.tensor([], dtype=torch.float)
        merged_multilabels = torch.tensor([], dtype=torch.float)

        for batch_idx, (images, multilabels) in enumerate(dataloader):
            images = images.cuda()
            multilabels = multilabels.cuda()

            logits = model(images)

            merged_logits = torch.cat([merged_logits, logits.cpu()])
            merged_multilabels = torch.cat([merged_multilabels, multilabels.cpu()])

        _, sorted_indexes = merged_logits.sort(dim=0, descending=True)

        aps = []
        for sorted_index, merged_multilabel in zip(sorted_indexes.t(), merged_multilabels.t()):
            merged_multilabel = merged_multilabel[sorted_index]
            ap = Metrics.interpolated_average_precision_at_n_points(merged_multilabel.numpy(), 11)
            aps.append(ap)
        aps = np.array(aps)

        mean_ap = np.mean(aps).item()
        print(f'mean AP = {mean_ap:.4f}')

    with open(os.path.join(path_to_results_dir, 'mean_ap.txt'), 'w') as fp:
        fp.write(f'{mean_ap:.4f}')

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g.: ./checkpoints/model-100.pth')
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-r', '--results_dir', default='./results', help='path to results directory')
        args = parser.parse_args()

        path_to_checkpoint = args.checkpoint
        path_to_data_dir = args.data_dir
        path_to_results_dir = args.results_dir

        _eval(path_to_checkpoint, path_to_data_dir, path_to_results_dir)

    main()
