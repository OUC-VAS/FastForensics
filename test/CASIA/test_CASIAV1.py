from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np
from model.fastforiens import EiffVit_seg
import os
import torch
import torchvision.transforms
from torchvision.transforms import transforms, ToPILImage

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from PIL import Image

device = torch.device('cuda:0')

fileList = []
model = EiffVit_seg(num_classes=2)
checkpoint = torch.load('../checkpoint/CASIA/CASIA.pth', map_location='cpu')
msg = model.load_state_dict(checkpoint, strict=False)
print(msg)


def calculate_eer_f1(seg_map_flatten, logit_flatten):
    fpr, tpr, threshold = roc_curve(seg_map_flatten, logit_flatten, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    pred_eer = np.where(logit_flatten > eer_threshold, 1, 0)
    f1_eer = f1_score(seg_map_flatten, pred_eer)
    return f1_eer


# Please put the CASIAV1 dataset path in fileList
for file in os.listdir():
    fileList.append()


data_transform = torchvision.transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
mask_transform = torchvision.transforms.Compose([
            transforms.Resize((224, 224)),
        ])


acc = 0
val_f1 = 0
val_best_f1 = 0
val_auc = 0
val_eer = 0
model.cuda()
model.eval()
with torch.no_grad():
    for i_path in fileList:
        image = Image.open(i_path).convert('RGB')
        image = data_transform(image).unsqueeze(dim=0).to('cuda')

        mask = Image.open(i_path).convert('L')
        mask = mask_transform(mask)
        mask = np.asarray(mask).astype(np.float32) / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = torch.from_numpy(mask).to('cuda')

        output = model(image)

        one_image = output
        one_image = torch.softmax(one_image, dim=1)[:, 1, :, :]

        # ACC & F1
        result_mask = np.where(one_image.cpu().detach().numpy() > 0.5, 1, 0).flatten()
        gt_mask = mask.cpu().detach().numpy().flatten()
        acc += (result_mask == gt_mask).sum() / result_mask.size
        val_f1 += f1_score(gt_mask, result_mask)

        # AUC
        result_mask = one_image.cpu().detach().numpy().flatten()
        gt_mask = mask.cpu().detach().numpy().flatten()
        val_auc += roc_auc_score(gt_mask, result_mask)

        # Best-F1
        result_mask = one_image.cpu().detach().numpy().flatten()
        gt_mask = mask.cpu().detach().numpy().flatten()
        val_eer += calculate_eer_f1(gt_mask, result_mask)
        precisions, recalls, thresholds = precision_recall_curve(gt_mask, result_mask)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        val_best_f1 += np.max(f1_scores[np.isfinite(f1_scores)])


print('AUC:%3.6f F1:%3.6f B_F1:%3.6f ERR:%3.6f ACC:%3.6f' % \
      ((val_auc / 920), (val_f1 / 920),(val_best_f1 / 920), (val_eer / 920), (acc / 920)))

