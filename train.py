import torch
import os
from dataset_mat2 import WHU_OHS_Dataset
from nets.DTSU import DTSU, Encoder,Decoder
import torch.optim as optim
from tqdm import tqdm


import time

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

def main():
    print('Build model ...')
    model_name = 'DTSU'

    config = dict(
        in_channels=32,
        num_classes=24,
        block_channels=(96, 128, 192, 256, 320, 384),
        num_blocks=(1, 1, 1, 1, 1, 1),
        inner_dim=128,
        reduction_ratio=1.0,
    )


    model = DTSU(encoder=Encoder,decoder=Decoder).to('cuda:0')



    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    epoch_num = 100
    batch_size = 2
    print('Load data ...')

    data_root = './data/'
    # data_root = '/hy-tmp/'

    # Choose which image to use for training
    image_prefix = 'T'

    data_path_train_image = os.path.join(data_root, 'tr', 'image')
    data_path_val_image = os.path.join(data_root, 'val', 'image')

    train_image_list = []
    train_label_list = []
    val_image_list = []
    val_label_list = []

    for root, paths, fnames in sorted(os.walk(data_path_train_image)):
        for fname in fnames:
            if is_image_file(fname):
                # if ((image_prefix + '_') in fname):
                if 'T'  in fname:
                    image_path = os.path.join(data_path_train_image, fname)
                    label_path = image_path.replace('image', 'label')
                    assert os.path.exists(label_path)
                    assert os.path.exists(image_path)
                    train_image_list.append(image_path)
                    train_label_list.append(label_path)

    for root, paths, fnames in sorted(os.walk(data_path_val_image)):
        for fname in fnames:
            if is_image_file(fname):
            #     if ((image_prefix + '_') in fname):
               if 'T'  in fname:
                    image_path = os.path.join(data_path_val_image, fname)
                    label_path = image_path.replace('image', 'label')
                    assert os.path.exists(label_path)
                    assert os.path.exists(image_path)
                    val_image_list.append(image_path)
                    val_label_list.append(label_path)

    assert len(train_image_list) == len(train_label_list)
    assert len(val_image_list) == len(val_label_list)

    train_dataset = WHU_OHS_Dataset(image_file_list=train_image_list, label_file_list=train_label_list)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    val_dataset = WHU_OHS_Dataset(image_file_list=val_image_list, label_file_list=val_label_list)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)


    lambda_lr = lambda x: (1 - x / epoch_num) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    model_path = './model/' + image_prefix + '_' + model_name + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print('Start training.')
    min_val_loss = 999
    start = time.time()

    for epoch in range(epoch_num):
        print('Epoch: %d/%d' % (epoch + 1, epoch_num))
        print('Current learning rate: %.8f' % (optimizer.state_dict()['param_groups'][0]['lr']))

        model.train()
        batch_index = 0
        loss_sum = 0

        for data, label, _ in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            pred = model(data)
            main_loss = criterion(pred, label)

            loss = main_loss
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss.item()
            batch_index = batch_index + 1
            average_loss_cur = loss_sum / batch_index
            if (batch_index % 5 == 0):
                print('training loss %.6f' % (average_loss_cur))

        average_loss = loss_sum / batch_index
        print('Epoch [%d/%d] training loss %.6f' % (epoch + 1, epoch_num, average_loss))

        with torch.no_grad():
            model.eval()
            val_index = 0
            val_loss_sum = 0

            for data, label, _ in tqdm(val_loader):
                data = data.to(device)
                label = label.to(device)

                pred = model(data)
                val_loss = criterion(pred, label)
                val_loss_sum = val_loss_sum + val_loss
                val_index = val_index + 1

            average_val_loss = val_loss_sum / val_index
            print('Epoch [%d/%d] validation loss %.6f' % (epoch + 1, epoch_num, average_val_loss))

            # if(average_val_loss < min_val_loss):
            #     min_val_loss = average_val_loss
            #     # Update the best model evaluated on the validation set
            #     torch.save(model.state_dict(), model_path + model_name + '_update_' + str(epoch) + '.pth')

        # Save model regularly
        if (epoch % 10 == 0):
            torch.save(model.state_dict(), model_path + model_name + '_' + str(epoch) + '.pth')

        scheduler.step()

    # Save model for the final epoch
    torch.save(model.state_dict(), model_path + model_name + '_final.pth')
    end = time.time()
    print("程序的运行时间为：{}".format(end - start))

if __name__ == '__main__':
    main()



