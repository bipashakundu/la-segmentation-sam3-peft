import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import models
import utils
from torchvision.utils import save_image
import torch.nn.functional as F

from torchvision import transforms


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, save_path=None):
    model.eval()
    
    data_norm = {
        'inp': {'sub': [0.485, 0.456, 0.406], 'div': [0.229, 0.224, 0.225]},
        'gt': {'sub': [0], 'div': [1]}
    }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'kvasir':
        metric_fn = utils.calc_kvasir
        metric1, metric2, metric3, metric4 = 'dice', 'iou', 'none', 'none'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    # accumulate per-batch scalars — no large tensors stored
    all_dice    = []   # batch-level dice (for mean + std)
    all_iou     = []   # batch-level iou  (for mean + std)
    all_per_iou = []   # per-sample iou scalars (for mAP)

    pbar = tqdm(loader, leave=False, desc='val')

    if save_path is not None:
        path_img = os.path.join(save_path, 'image')
        path_gt = os.path.join(save_path, 'gt')
        path_pred = os.path.join(save_path, 'pred')
        os.makedirs(path_img, exist_ok=True)
        os.makedirs(path_gt, exist_ok=True)
        os.makedirs(path_pred, exist_ok=True)

    cnt = 0
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()

        inp = batch['inp']
        gt = batch['gt']

        with torch.no_grad():
            pred_logits = model.infer(inp)
            pred_prob = torch.sigmoid(pred_logits)

        result1, result2, result3, result4 = metric_fn(pred_prob, gt)
        val_metric1.add(float(result1), inp.shape[0])
        val_metric2.add(float(result2), inp.shape[0])

        # collect batch-level scalars for std; per-sample IoU for mAP
        all_dice.append(float(result1))
        all_iou.append(float(result2))
        _, iou_per = utils.calc_per_sample_metrics(pred_prob, gt)
        all_per_iou.extend(iou_per.tolist())
        try:
            val_metric3.add(float(result3), inp.shape[0])
            val_metric4.add(float(result4), inp.shape[0])
        except:
            pass

        if save_path is not None:
            batch_size = inp.shape[0]
            
            sub = torch.tensor(data_norm['inp']['sub']).cuda().view(-1, 1, 1)
            div = torch.tensor(data_norm['inp']['div']).cuda().view(-1, 1, 1)

            for i in range(batch_size):
                if 'name' in batch:
                    name = batch['name'][i]
                    name = os.path.splitext(name)[0]
                else:
                    name = f"{cnt}_{i}"
                
                target_h, target_w = inp.shape[2], inp.shape[3]
                
                if 'shape' in batch:
                     if isinstance(batch['shape'], list):
                         target_h = int(batch['shape'][0][i])
                         target_w = int(batch['shape'][1][i])
                     else:
                         target_h = int(batch['shape'][i][0])
                         target_w = int(batch['shape'][i][1])

                img_vis = inp[i].clone()
                img_vis = img_vis * div + sub
                img_vis = torch.clamp(img_vis, 0, 1)

                pred_vis = pred_prob[i:i+1].clone() # 保持 [1, H, W]
                gt_vis = gt[i:i+1].clone()          # 保持 [1, H, W]

                if target_h != img_vis.shape[1] or target_w != img_vis.shape[2]:
                    img_vis = F.interpolate(img_vis.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)

                    pred_vis = F.interpolate(pred_vis, size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
                    
                    gt_vis = F.interpolate(gt_vis, size=(target_h, target_w), mode='nearest').squeeze(0)
                
                else:
                     pred_vis = pred_vis.squeeze(0)
                     gt_vis = gt_vis.squeeze(0)

                     
                save_image(img_vis, os.path.join(path_img, f"{name}.png"))
                save_image(pred_vis, os.path.join(path_pred, f"{name}.png"))
                save_image(gt_vis, os.path.join(path_gt, f"{name}.png"))

            cnt += 1

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))

    import numpy as np
    per_iou_arr = np.array(all_per_iou)
    thresholds   = np.arange(0.50, 1.00, 0.05)
    aps = [np.sum(per_iou_arr >= t) / (len(per_iou_arr) + 1e-6) for t in thresholds]
    mAP  = float(np.mean(aps))
    ap50 = float(aps[0])
    ap75 = float(aps[5])

    print(f'\nmAP@0.50:0.95 = {mAP:.4f}  |  AP@0.50 = {ap50:.4f}  |  AP@0.75 = {ap75:.4f}')
    print(f'Dice: {val_metric1.item():.4f} ± {float(np.std(all_dice)):.4f}')
    print(f'IoU:  {val_metric2.item():.4f} ± {float(np.std(all_iou)):.4f}')

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/cod-sam-vit-l.yaml", help="配置文件路径")
    parser.add_argument('--model', required=True, help="训练好的 .pth 模型路径")
    parser.add_argument('--save_path', default=None, help="如果不为空，则将预测结果图片保存到该文件夹")
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8, shuffle=False)

    model = models.make(config['model']).cuda()
    
    print(f"Loading checkpoint from {args.model}...")
    checkpoint = torch.load(args.model, map_location='cuda:0')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[Info] Missing keys (adapter/unused layers, expected for pre-adapter checkpoints): {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {unexpected}")
    print("Model loaded successfully.")

    metric1, metric2, metric3, metric4 = eval_psnr(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   verbose=True,
                                                   save_path=args.save_path) # 传入保存路径

    print('################ Results ################')
    print(f'Dice: {metric1:.4f}')
    print(f'IoU:  {metric2:.4f}')
    print('#########################################')
