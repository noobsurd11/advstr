import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from haze import SimpleGridNet, TransmissionNet, FogGenerator
from dataloader import PairedOHazeDataset
from utils.losses import PerceptualLoss, dark_channel_loss
import torch.nn.functional as F

def vertical_gradient_tensor(B, H, W, device):
    Gy = torch.linspace(0,1,steps=H, device=device).view(1,1,H,1).expand(B,1,H,W)
    return Gy

def validate(loader, fog_gen, trans_net, percep, cfg, device):
    fog_gen.eval()
    trans_net.eval()

    total_l1 = 0
    total_lp = 0
    total_ld = 0
    count = 0

    with torch.no_grad():
        for J, I_gt, _ in loader:
            J = J.to(device)
            I_gt = I_gt.to(device)
            B,C,H,W = J.shape

            t = trans_net(J)
            Gy = vertical_gradient_tensor(B, H, W, device)
            t_mod = t * (1 - 0.4 * Gy)

            I_pred, fog_feat, A = fog_gen(J, t_mod)


            l1 = F.l1_loss(I_pred, I_gt)
            lp = percep(I_pred, I_gt)
            ld = dark_channel_loss(I_pred, I_gt)

            total_l1 += l1.item()
            total_lp += lp.item()
            total_ld += ld.item()
            count += 1

    return {
        'l1': total_l1 / count,
        'perc': total_lp / count,
        'dark': total_ld / count,
        'total': total_l1/count + cfg['loss']['perceptual_w']*(total_lp/count) + cfg['loss']['dark_w']*(total_ld/count)
    }

def train_loop(cfg):
    device = torch.device(cfg.get('device','cuda') if torch.cuda.is_available() else 'cpu')

    # Datasets
    train_ds = PairedOHazeDataset(cfg['data']['train_dir'], split='train', size=512)
    val_ds   = PairedOHazeDataset(cfg['data']['train_dir'], split='val', size=512)

    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Models
    backbone = SimpleGridNet(in_ch=3, base_ch=16, rdb_layers=4)
    trans_net = TransmissionNet()
    fog_gen = FogGenerator(backbone, alpha=cfg['model']['alpha'], beta_init=cfg['model']['beta_init'])

    fog_gen = fog_gen.to(device)
    trans_net = trans_net.to(device)

    percep = PerceptualLoss(layer_idx=16, device=device).to(device)


    # Optimizer
    opt = torch.optim.Adam(list(fog_gen.parameters()) + list(trans_net.parameters()), lr=cfg['train']['lr'])

    # Training Loop
    best_val = float('inf')
    epochs = cfg['train']['epochs']
    out_dir = cfg['log']['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        fog_gen.train()
        trans_net.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        running_loss = 0

        for J, I_gt, fn in pbar:
            J = J.to(device)
            I_gt = I_gt.to(device)
            B,C,H,W = J.shape

            # Forward
            t = trans_net(J)
            Gy = vertical_gradient_tensor(B, H, W, device)
            t_mod = t * (1 - 0.4 * Gy)

            I_pred, fog_feat, A = fog_gen(J, t_mod)

            # Loss
            l1 = F.l1_loss(I_pred, I_gt)
            lp = percep(I_pred, I_gt)
            ld = dark_channel_loss(I_pred, I_gt)

            loss = l1 + cfg['loss']['perceptual_w'] * lp + cfg['loss']['dark_w'] * ld

            # Backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            pbar.set_postfix(train_loss=f"{running_loss/(pbar.n+1):.5f}")

        # Validation
        val_metrics = validate(val_loader, fog_gen, trans_net, percep, cfg, device)
        print(f"Validation: L1={val_metrics['l1']:.4f} Perc={val_metrics['perc']:.4f} "
              f"Dark={val_metrics['dark']:.4f} Total={val_metrics['total']:.4f}")

        # Save best model
        if val_metrics['total'] < best_val:
            best_val = val_metrics['total']
            torch.save({
                'epoch': epoch,
                'fog_gen': fog_gen.state_dict(),
                'trans_net': trans_net.state_dict(),
                'opt': opt.state_dict(),
                'val_loss': best_val,
            }, os.path.join(out_dir, f"best_model.pth"))
            print(f"Saved new BEST checkpoint (epoch {epoch}).")

        # Save periodic checkpoints
        if epoch % cfg['train']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'fog_gen': fog_gen.state_dict(),
                'trans_net': trans_net.state_dict(),
                'opt': opt.state_dict(),
            }, os.path.join(out_dir, f"checkpoint_ep{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/train.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    train_loop(cfg)
