from tqdm import tqdm
from dataset.cad_dataset import get_dataloader
from config import ConfigAE
from utils import ensure_dir
from trainer import TrainerAE,TrainerGSEncoder
import torch
import numpy as np
import os
import h5py
from cadlib.macro import EOS_IDX


def main():
    # create experiment cfg containing all hyperparameters
    cfg = ConfigAE('train')
    encode(cfg)
def encode(cfg):
    # create network and training agent
    tr_agent = TrainerGSEncoder(cfg)

    # # load from checkpoint if provided
    # tr_agent.load_ckpt(cfg.ckpt)
    # tr_agent.net.train()

    # create dataloader
    save_dir = "{}/results".format(cfg.exp_dir)
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, 'all_zs_ckpt{}.h5'.format(cfg.ckpt))
    fp = h5py.File(save_path, 'w')
    for phase in ['train', 'validation', 'test']:
        train_loader = get_dataloader(phase, cfg, shuffle=False)

        # encode
        all_zs = []
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            # with torch.no_grad():
                z = tr_agent.encode(data, is_batch=True)
                z = z.detach().cpu().numpy()[:, 0, :]
                all_zs.append(z)
        all_zs = np.concatenate(all_zs, axis=0)
        print(all_zs.shape)
        fp.create_dataset('{}_zs'.format(phase), data=all_zs)
    fp.close()


if __name__ == '__main__':
    main()
