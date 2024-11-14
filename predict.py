import torch
import os
import numpy as np
import torch
import os
import numpy as np
# from osgeo import gdal
from nets.shiyan import FreeNet
from PIL import Image
from dataset_mat2 import WHU_OHS_Dataset
# from FreeNet import FreeNet
# from nets.unetformer import UNetFormer
# from nets.hrnet import HRnet
from tqdm import tqdm
# from nets.my4 import DAT, Encoder
# from nets.unet import Unet
# from nets.my4 import DAT, Encoder
from nets.BEDSN import DAT, Encoder, Decoder
# from nets.My import FreeNet
# from nets.mymy import FreeNet
# from nets.xiaorong import FreeNet
# from torchsummary import summary
from tqdm import tqdm
# import nets.config as configs
# from nets.STUNet1 import VisionTransformer


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def writeTiff(im_data, im_width, im_height, im_bands, path):
    if len(im_data.shape) == 2:
        im_data = np.expand_dims(im_data, axis=-1)  # 将 2D 数组转为 3D（高 x 宽 x 通道）

    im = Image.fromarray(im_data.squeeze(), mode='L')  # 'L' 表示单通道灰度图像
    im.save(path)
# def writeTiff(im_data, im_width, im_height, im_bands, path):
#     if 'int8' in im_data.dtype.name:
#         datatype = gdal.GDT_Byte
#     elif 'int16' in im_data.dtype.name:
#         datatype = gdal.GDT_UInt16
#     else:
#         datatype = gdal.GDT_Float32
#
#     if len(im_data.shape) == 3:
#         im_bands, im_height, im_width = im_data.shape
#     elif len(im_data.shape) == 2:
#         im_data = np.array([im_data])
#     else:
#         im_bands, (im_height, im_width) = 1, im_data.shape
#
#     driver = gdal.GetDriverByName("GTiff")
#     dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
#
#     for i in range(im_bands):
#         dataset.GetRasterBand(i+1).WriteArray(im_data[i])
#     del dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def main():
    print('Build model ...')
    model_name = 'DTSU'

    config = dict(
        in_channels=32,
        num_classes=24,
        block_channels=(96, 128, 192, 256, 320, 384),
        num_blocks=(1, 1, 1, 1, 1, 1),
        # block_channels=(96, 128, 192, 256),
        # num_blocks=(1, 1, 1, 1),
        inner_dim=256,
        reduction_ratio=1.0,
    )
    # model = FCN(encoder_name='resnet50', in_channel_nb=32, classes_nb=24).to('cuda:0')
    # model = DAT(encoder=Encoder).to('cuda:0')
    model = DAT(encoder=Encoder,decoder=Decoder).to('cuda:0')
    # model = LANet().to('cuda:0')
    # model = VisionTransformer(configs.get_r50_b16_config()).to('cuda:0')
    # model   = DeepLab(num_classes=24, backbone='resnet50', pretrained=False).to(device)
    # model   = emrt().to(device)
    # model = FreeNet(config).to(device)
    # model   = Unet(num_classes=24, backbone='resnet50', pretrained=False).to(device)
    # model=DANet().to('cuda:0')
    # model = MAResUNet(32).to('cuda:0')
    # model = Segformer(
    #     dims = (32, 64, 160, 256),      # dimensions of each stage
    #     heads = (1, 2, 5, 8),           # heads of each stage
    #     ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    #     reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    #     num_layers = 2,                 # num layers of each stage
    #     decoder_dim = 256,              # decoder dimension
    #     num_classes = 2                 # number of segmentation classes
    # ).to('cuda:0')
    # model =  DCSwin(encoder_channels=(96, 192, 384, 768),
    #                num_classes=24,
    #                embed_dim=96,
    #                depths=(2, 2, 6, 2),
    #                num_heads=(3, 6, 12, 24),
    #                frozen_stages=2).to('cuda:0')
    # model = UNetFormer(decode_channels=64,
    #                    dropout=0.1,
    #                    backbone_name='swsl_resnet18',
    #                    pretrained=False,
    #                    window_size=8,
    #                    num_classes=24).to(device)
    # model = CNN_3D(input_features=32, n_classes=24).to(device)
    # model = A2S2KResNet(band=32, classes=24, reduction=2).to(device)

    # summary(model, (32, 512, 512))

    # Load model (model of final epoch or best model evaluated on the validation set)
    model_path = './model/edge_100.pth'
    model.load_state_dict(torch.load(model_path))
    print('Loaded trained model.')

    print('Load data ...')
    data_root = '/hy-tmp/'
    image_prefix = 'T5'

    data_path_test_image = os.path.join(data_root, 'ts', 'image')

    test_image_list = []
    test_label_list = []

    for root, paths, fnames in sorted(os.walk(data_path_test_image)):
        for fname in fnames:
            # if is_image_file(fname):
            #     if ((image_prefix + '_') in fname):
                    image_path = os.path.join(data_path_test_image, fname)
                    label_path = image_path.replace('image', 'label')
                    assert os.path.exists(label_path)
                    assert os.path.exists(image_path)
                    test_image_list.append(image_path)
                    test_label_list.append(label_path)

    assert len(test_image_list) == len(test_label_list)

    test_dataset = WHU_OHS_Dataset(image_file_list=test_image_list, label_file_list=test_label_list)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    print('Predicting.')

    save_path = './result/' + image_prefix + '_' + model_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        model.eval()
        for data, _, name in tqdm(test_loader):
            data = data.to(device)
            pred,edge = model(data)
            output = pred[0, :, :, :].argmax(axis=0)
            output = output.cpu().detach().numpy() + 1
            output = output.astype(np.uint8)
            name[0] = name[0].replace('.mat', '.tif')
            writeTiff(output, 512, 512, 1, os.path.join(save_path, name[0]))

if __name__ == '__main__':
    main()




