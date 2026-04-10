# https://docs.lightly.ai/self-supervised-learning/examples/mae.html
import os
import torch
import torchvision
from torch import nn

from lightly.models import utils
from lightly.models.modules import masked_autoencoder
from lightly.transforms.mae_transform import MAETransform
from lightly.utils.debug import std_of_l2_normalized
from lightly.data.dataset import LightlyDataset
import numpy as np

#############
# edit
exp_id = 'MAE_dfv2_0'
exp_path = f'./exps'
if not os.path.exists(exp_path): os.makedirs(exp_path)
save_path = f'{exp_path}/{exp_id}'
if not os.path.exists(save_path): os.makedirs(save_path)
save_model_path = f'{save_path}/model' # entire model
if not os.path.exists(save_model_path): os.makedirs(save_model_path)
save_bb_path = f'{save_path}/bb' # backbone
if not os.path.exists(save_bb_path): os.makedirs(save_bb_path)
save_dec_path = f'{save_path}/proj' # decoder
if not os.path.exists(save_dec_path): os.makedirs(save_dec_path)

log_path = f'{save_path}/log'
if not os.path.exists(log_path): os.makedirs(log_path)
log_epoch_file_path = f'{log_path}/epoch.log'
log_epoch_file = open(log_epoch_file_path, 'a')
log_epoch_file.write('epoch_i,avg_loss\n')

save_interval = 20
epochs = 300

dataset_path = '/media/brcao/eData2/Data/datasets/LAVN_dfv2'
#############

class MAE(nn.Module):
    def __init__(self, vit):
        super().__init__()

        decoder_dim = 512
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=vit.patch_size**2 * 3,
            dropout=0,
            attention_dropout=0,
        )

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        return x_pred, target


vit = torchvision.models.vit_b_32(pretrained=False)
model = MAE(vit)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = MAETransform()
dataset = LightlyDataset(dataset_path, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

print("Starting Training")
for epoch in range(epochs):
    total_loss, curr_loss = 0, float('inf')
    ##########################
    # Save path
    save_model_file_path = f'{save_model_path}/model_epoch_{epoch}.pth' # entire model
    save_bb_file_path = f'{save_bb_path}/bb_epoch_{epoch}.pth' # backbone
    save_dec_file_path = f'{save_dec_path}/dec_epoch_{epoch}.pth' # decoder

    save_model_best_file_path = f'{save_model_path}/model_best.pth' # entire model
    save_bb_best_file_path = f'{save_bb_path}/bb_best.pth' # backbone
    save_dec_best_file_path = f'{save_dec_path}/dec_best.pth' # decoder
    ##########################

    for i, batch in enumerate(dataloader):
        views = batch[0]
        images = views[0].to(device)  # views contains only a single view
        predictions, targets = model(images)
        loss = criterion(predictions, targets)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    log_epoch_file.write(f'{epoch},{avg_loss}\n')

    # Save model
    if epoch % epochs == 0:
        torch.save(model.state_dict(), save_model_file_path); print(f'{save_model_file_path} saved!')
        torch.save(model.backbone.state_dict(), save_bb_file_path); print(f'{save_bb_file_path} saved!')
        torch.save(model.decoder.state_dict(), save_dec_file_path); print(f'{save_dec_file_path} saved!')
    if avg_loss < curr_loss:
        torch.save(model.state_dict(), save_model_best_file_path); print(f'{save_model_best_file_path} saved!')
        torch.save(model.backbone.state_dict(), save_bb_best_file_path); print(f'{save_bb_best_file_path} saved!')
        torch.save(model.decoder.state_dict(), save_dec_best_file_path); print(f'{save_dec_best_file_path} saved!')
        curr_loss = avg_loss

    '''
    Load model
    model.load_state_dict(torch.load(save_model_file_path)); print(f'{save_model_file_path} loaded!')
    model.backbone.load_state_dict(torch.load(save_bb_file_path)); print(f'{save_bb_file_path} loaded!')
    model.decoder.load_state_dict(torch.load(save_dec_file_path)); print(f'{save_dec_file_path} loaded!')
    '''
