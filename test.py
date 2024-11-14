import torch
import os
import numpy as np
from nets.shiyan import FreeNet
from dataset_mat2 import WHU_OHS_Dataset
from nets.shiyan import FreeNet
from nets.my_self import DAT, Encoder,Decoder
# from nets.unetformer import UNetFormer
# from nets.my4 import FreeNet
# from nets.hrnet import HRnet
from tqdm import tqdm

# from nets.unet import Unet
# from nets.myxiaorong2 import DAT, Encoder
# from nets.My import FreeNet
# from nets.mynoconv import DAT, Encoder
# from nets.mymy import FreeNet
# from nets.xiaorong import FreeNet
# from torchsummary import summary

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif','mat'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def genConfusionMatrix(numClass, imgPredict, imgLabel):
    mask = (imgLabel != -1)
    label = numClass * imgLabel[mask] + imgPredict[mask]
    count = torch.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix

def main():
    print('Build model ...')
    model_name = 'My'

    config = dict(
        in_channels=32,
        num_classes=24,
        # block_channels=(96, 128, 192, 256),
        # block_channels=(96, 128, 192, 256, 512, 1024),
        block_channels=(96, 128, 192, 256, 320, 384),
        # num_blocks=(1, 1, 1, 1),
        num_blocks=(1, 1, 1, 1, 1, 1),
        inner_dim=128,
        reduction_ratio=1.0,
    )

    # model = DAT(encoder=Encoder).to('cuda:0')
    model = DAT(encoder=Encoder,decoder=Decoder).to('cuda:0')
    # model = FreeNet(config).to(device)
    # model   = HRnet(num_classes=24, backbone='hrnetv2_w18', pretrained=False).to(device)
    # model = UNetFormer(decode_channels=64,
    #                    dropout=0.1,
    #                    backbone_name='swsl_resnet18',
    #                    pretrained=False,
    #                    window_size=8,
    #                    num_classes=24).to(device)
    # model   = Unet(num_classes=24, backbone='resnet50', pretrained=False).to(device)

    # summary(model, (32, 512, 512))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Load model (model of final epoch or best model evaluated on the validation set)
    # model_path = './model/S_shiyan/shiyan_90.pth'
    model_path = './models/finetuned_model/S2HM2_seg_10.pth'
    # model_path = './model/T5/T5_unetformer/unetformer_80.pth'
    model.load_state_dict(torch.load(model_path))
    print('Loaded trained model.')

    print('Load data ...')
    data_root = './data/'
    # data_root = '/hy-tmp/'
    image_prefix = 'S1'

    data_path_test_image = os.path.join(data_root, 'val', 'image')

    test_image_list = []
    test_label_list = []

    for root, paths, fnames in sorted(os.walk(data_path_test_image)):
        for fname in fnames:
            if is_image_file(fname):
                if 'S'  in fname:
            #     if ((image_prefix + '_') in fname):
                    image_path = os.path.join(data_path_test_image, fname)
                    label_path = image_path.replace('image', 'label')
                    assert os.path.exists(label_path)
                    assert os.path.exists(image_path)
                    test_image_list.append(image_path)
                    test_label_list.append(label_path)

    assert len(test_image_list) == len(test_label_list)

    class_num = 24

    test_dataset = WHU_OHS_Dataset(image_file_list=test_image_list, label_file_list=test_label_list)
    # ,use_3D_input=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
                                              )

    print('Testing.')

    with torch.no_grad():
        model.eval()
        confusionmat = torch.zeros([class_num, class_num])
        confusionmat = confusionmat.to(device)

        for data, label, _ in tqdm(test_loader):
            data = data.to(device)
            label = label.to(device)

            # pred,edge = model(data)
            pred = model(data)
            output = pred[0, :, :, :].argmax(axis=0)
            label = label[0, :, :]

            confusionmat_tmp = genConfusionMatrix(class_num, output, label)
            confusionmat = confusionmat + confusionmat_tmp

    confusionmat = confusionmat.cpu().detach().numpy()

    unique_index = np.where(np.sum(confusionmat, axis=1) != 0)[0]
    # unique_index = np.arange(confusionmat.shape[0])
    confusionmat = confusionmat[unique_index, :]
    confusionmat = confusionmat[:, unique_index]

    a = np.diag(confusionmat)
    b = np.sum(confusionmat, axis=0)
    c = np.sum(confusionmat, axis=1)

    eps = 0.0000001

    PA = a / (c + eps)
    UA = a / (b + eps)
    print('PA:', PA)
    print('UA:', UA)

    F1 = 2 * PA * UA / (PA + UA + eps)
    print('F1:', F1)

    AA = np.nanmean(PA)
    print('AA:',AA)

    mean_F1 = np.nanmean(F1)
    print('mean F1:', mean_F1)

    OA = np.sum(a) / np.sum(confusionmat)
    print('OA:', OA)

    PE = np.sum(b * c) / (np.sum(c) * np.sum(c))
    Kappa = (OA - PE) / (1 - PE)
    print('Kappa:', Kappa)

    intersection = np.diag(confusionmat)
    union = np.sum(confusionmat, axis=1) + np.sum(confusionmat, axis=0) - np.diag(confusionmat)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)


    print('mIoU:', mIoU)
    print('IoU:', IoU)
    print(f"{mIoU*100:.2f}")
    print(f"{OA*100:.2f}")
    print(f"{mean_F1*100:.2f}")
    class_names = ["稻田", "干农田", "林地", "灌木", "稀疏林地", "其他林地", "高覆盖草地", "中等覆盖草地", "低覆盖草地", "河流",
                   "湖泊", "水库/池塘", "海岸", "浅滩", "城区", "农村", "其他建筑地", "沙地", "戈壁", "盐碱地",
                   "沼泽地", "裸地", "裸岩", "海洋"]
    for i, class_name in enumerate(class_names):
        # print(f"Class: {class_name}")
        # print(f"PA: {PA[i]}")
        # print(f"F1: {F1[i]}")
        print(f"{IoU[i]*100:.2f}")

if __name__ == '__main__':
    main()