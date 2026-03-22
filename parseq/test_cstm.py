import torch
from parseq.strhub.models.utils import load_from_checkpoint
from parseq.strhub.data.dataset import SceneTextDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

ckpt = "parseq/checkpoints/parseq_hazy.ckpt"
data_root = "parseq/data/test/hazy"

model = load_from_checkpoint(ckpt).eval().cuda()
hp = model.hparams

# The LMDB dataset
ds = SceneTextDataset(
    data_root,
    img_size=hp.img_size,
    max_label_length=hp.max_label_length,
    charset=hp.charset_test,
    augment=False
)
dl = DataLoader(ds, batch_size=1, num_workers=4)

total = 0
correct = 0

for imgs, labels in tqdm(dl):
    imgs = imgs.cuda()
    out = model.test_step((imgs, labels), -1)['output']
    total += out.num_samples
    correct += out.correct

print("Accuracy:", 100 * correct / total)
