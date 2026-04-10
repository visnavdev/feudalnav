# https://docs.lightly.ai/self-supervised-learning/examples/simclr.html
import os
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.utils.debug import std_of_l2_normalized
from lightly.data.dataset import LightlyDataset
import numpy as np

#############
# edit
exp_id = 'SimCLR_dfv2_0'
exp_path = f'./exps'
if not os.path.exists(exp_path): os.makedirs(exp_path)
save_path = f'{exp_path}/{exp_id}'
if not os.path.exists(save_path): os.makedirs(save_path)
save_model_path = f'{save_path}/model' # entire model
if not os.path.exists(save_model_path): os.makedirs(save_model_path)
save_bb_path = f'{save_path}/bb' # backbone
if not os.path.exists(save_bb_path): os.makedirs(save_bb_path)
save_proj_path = f'{save_path}/proj' # projection
if not os.path.exists(save_proj_path): os.makedirs(save_proj_path)

log_path = f'{save_path}/log'
if not os.path.exists(log_path): os.makedirs(log_path)
log_batch_file_path = f'{log_path}/batch.log'
log_batch_file = open(log_batch_file_path, 'a')
log_batch_file.write('epoch_i,batch_i,std_l2_0,std_l2_1\n')

log_epoch_file_path = f'{log_path}/epoch.log'
log_epoch_file = open(log_epoch_file_path, 'a')
log_epoch_file.write('epoch_i,avg_loss\n')

save_interval = 20
epochs = 300

dataset_path = '/media/brcao/eData2/Data/datasets/LAVN_dfv2'
#############

class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimCLR(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
dataset = LightlyDataset(dataset_path, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(epochs):
    total_loss, curr_loss = 0, float('inf')
    ##########################
    # Save path
    save_model_file_path = f'{save_model_path}/model_epoch_{epoch}.pth' # entire model
    save_bb_file_path = f'{save_bb_path}/bb_epoch_{epoch}.pth' # backbone
    save_proj_file_path = f'{save_proj_path}/proj_epoch_{epoch}.pth' # projection

    save_model_best_file_path = f'{save_model_path}/model_best.pth' # entire model
    save_bb_best_file_path = f'{save_bb_path}/bb_best.pth' # backbone
    save_proj_best_file_path = f'{save_proj_path}/proj_best.pth' # projection
    ##########################

    for i, batch in enumerate(dataloader):
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Use std_of_l2_normalized to check if the representations collapse
        std_l2_0, std_l2_1 = std_of_l2_normalized(z0), std_of_l2_normalized(z1)
        # print('\nstd_l2_0: ', std_l2_0, ', std_l2_1: ', std_l2_1)
        log_batch_file.write(f'{i},{std_l2_0},{std_l2_1}\n')
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    log_epoch_file.write(f'{epoch},{avg_loss}\n')

    # Save model
    if epoch % epochs == 0:
        torch.save(model.state_dict(), save_model_file_path); print(f'{save_model_file_path} saved!')
        torch.save(model.backbone.state_dict(), save_bb_file_path); print(f'{save_bb_file_path} saved!')
        torch.save(model.projection_head.state_dict(), save_proj_file_path); print(f'{save_proj_file_path} saved!')
    if avg_loss < curr_loss:
        torch.save(model.state_dict(), save_model_best_file_path); print(f'{save_model_best_file_path} saved!')
        torch.save(model.backbone.state_dict(), save_bb_best_file_path); print(f'{save_bb_best_file_path} saved!')
        torch.save(model.projection_head.state_dict(), save_proj_best_file_path); print(f'{save_proj_best_file_path} saved!')
        curr_loss = avg_loss

    '''
    Load model
    model.load_state_dict(torch.load(save_model_file_path)); print(f'{save_model_file_path} loaded!')
    model.backbone.load_state_dict(torch.load(save_bb_file_path)); print(f'{save_bb_file_path} loaded!')
    model.projection_head.load_state_dict(torch.load(save_proj_file_path)); print(f'{save_proj_file_path} loaded!')
    '''