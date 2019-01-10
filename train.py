import argparse
import os
import time
from collections import deque

from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model


def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    dataset = Dataset(path_to_data_dir, Dataset.Mode.TRAIN)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    model = Model().cuda()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=70000, gamma=0.1)

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False

    num_steps_to_display = 20
    num_steps_to_snapshot = 10000
    num_steps_to_finish = 100000

    print('Start training')

    while not should_stop:
        for batch_idx, (images, multilabels) in enumerate(dataloader):
            images = images.cuda()
            multilabels = multilabels.cuda().float()

            logits = model.train().forward(images)
            loss = model.loss(logits, multilabels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1

            if step == num_steps_to_finish:
                should_stop = True

            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                avg_loss = sum(losses) / len(losses)
                lr = scheduler.get_lr()[0]
                print(f'[Step {step}] Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr} ({steps_per_sec:.2f} steps/sec)')

            if step % num_steps_to_snapshot == 0 or should_stop:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print(f'Model has been saved to {path_to_checkpoint}')

            if should_stop:
                break

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        _train(path_to_data_dir, path_to_checkpoints_dir)

    main()
