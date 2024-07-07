import os
import yaml
import torch
import argparse
import numpy as np
from attrdict import AttrDict
from collections import OrderedDict
from tqdm import tqdm
from predict_provider import Provider_valid
from model.encoder import UNet_PNI_encoder
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='pretraining_snemi3d', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default='pretrained_snemi3d')
    parser.add_argument('-id', '--model_id', type=str, default='ac43/model-200000')
    parser.add_argument('-m', '--mode', type=str, default='')
    parser.add_argument('-s', '--save', action='store_false', default=True)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name
    out_path = os.path.join('./inference', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'feat_' + args.model_id
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)
    device = torch.device('cuda:0')

    print('load superhuman model!')
    model = UNet_PNI_encoder(in_planes=cfg.MODEL.input_nc,
                             filters=cfg.MODEL.filters,
                             pad_mode=cfg.MODEL.pad_mode,
                             bn_mode=cfg.MODEL.bn_mode,
                             relu_mode=cfg.MODEL.relu_mode,
                             init_mode=cfg.MODEL.init_mode, ).to(device)

    ckpt_path = os.path.join('./trained_model', trained_model, args.model_id + '.ckpt')
    print(ckpt_path)
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k.replace('module.', '') if 'module' in k else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)

    valid_provider = Provider_valid(cfg)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)
    model.eval()

    print('the number of sub-volume:', len(valid_provider))
    losses_valid = []
    pbar = tqdm(total=len(valid_provider))

    feature_list = []
    position_list = []
    for kk, data in enumerate(val_loader, 0):
        inputs = data
        inputs = inputs.cuda()
        with torch.no_grad():
            representation = model(inputs)
            feature_list.append(representation.cpu().numpy())
            position_list.append(np.array(valid_provider.pos))
        pbar.update(1)
    pbar.close()

    np.save(os.path.join(out_affs, 'features.npy'), feature_list)
    np.save(os.path.join(out_affs, 'position.npy'), position_list)
