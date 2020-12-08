"""main body for theme rendering"""
import argparse
import os
import sys
import time
import numpy as np
import random
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import utils
from Pastiche import PasticheModel
from feature_extractor import vgg11
from PIL import Image

manual_seed = 8888
train_size = (480, 640)
eval_size = (1080, 810)
content_dataset_path = "coco-2017"
theme_dataset_path = "filter_images"
log_interval = 50
subset_size = 5000
total_filters = 9

Mean = [0.5, 0.5, 0.5]
Std = [0.2, 0.2, 0.2]


def batch_norm(batch):
    mean = batch.new_tensor(Mean).view(-1, 1, 1)
    std = batch.new_tensor(Std).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def train(args, device):
    # original degree (degree=2.2) Above 2.4 is recommended
    L_c = 10 ** 5  # content loss weight
    L_s = 10 ** 8.5  # theme loss weight

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    content_transform = transforms.Compose([
        transforms.Resize(train_size),
        # transforms.CenterCrop(train_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_dataset = datasets.ImageFolder(content_dataset_path, content_transform)
    index_list = list(range(len(content_dataset)))
    subset_list = random.sample(index_list, subset_size)
    content_dataset = torch.utils.data.Subset(content_dataset, subset_list)
    # print(len(content_dataset))
    theme_dataset = [img for img in os.listdir(theme_dataset_path)]
    # sort on filter index
    theme_dataset = sorted(theme_dataset, key=lambda i: i[-5])
    # print(theme_dataset)

    train_loader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True)
    num_themes = len(theme_dataset)
    global total_filters
    total_filters = num_themes

    PM = PasticheModel(num_themes).to(device)
    optimizer = Adam(PM.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = vgg11().to(device)
    theme_transform = transforms.Compose([
        transforms.Resize(train_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    theme_batch = []

    for i in range(num_themes):
        theme = Image.open(theme_dataset_path + '/' + theme_dataset[i])
        theme = theme_transform(theme)
        theme_batch.append(theme)

    themes = torch.stack(theme_batch).to(device)

    theme_features = vgg(batch_norm(themes))
    theme_gram = [gram_matrix(i) for i in theme_features]
    # degree of the filtering we want to apply
    degree = args.filtering_level

    if degree <= 0:
        L_s = 1
    else:
        L_s = L_s * 10 ** min(degree - 1, 5)

    for epoch in range(args.epochs):
        PM.train()
        count = 0

        for batch_idx, (x, _) in enumerate(train_loader):

            if len(x) < args.batch_size:
                break

            count += len(x)
            optimizer.zero_grad()

            theme_ids = []
            # random indices sampling to prepare for expansion
            for i in range(len(x)):
                id = random.randint(0, num_themes - 1)
                theme_ids.append(id)

            stylized = PM(x.to(device), theme_ids)

            stylized = batch_norm(stylized)
            contents = batch_norm(x)

            features_stylized = vgg(stylized.to(device))
            features_contents = vgg(contents.to(device))

            # use the last block to last block to compute high-level content loss
            # content_loss = mse_loss(features_stylized[-1], features_contents[-1])
            # use second to last block to compute high-level content loss
            content_loss = mse_loss(features_stylized[-2], features_contents[-2])
            content_loss = L_c * content_loss

            theme_loss = 0

            for ft_y, s_gram in zip(features_stylized, theme_gram):
                y_gram = gram_matrix(ft_y)
                theme_loss += mse_loss(y_gram, s_gram[theme_ids, :, :])
            theme_loss = L_s * theme_loss

            total_loss = content_loss + theme_loss
            total_loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print("Epoch {}:\t[{}/{}]\tcontent loss: {:.4f}\ttheme loss: {:.4f}\ttotal loss: {:.4f}".format(
                    epoch + 1, batch_idx, len(train_loader),
                    content_loss / (batch_idx + 1),
                    theme_loss / (batch_idx + 1),
                    (content_loss + theme_loss) / (batch_idx + 1)
                ))

    # save model
    saved_as = '{0}/{1}_filter_level_{2}_epoch{3}.pth'.format(args.save, str(time.ctime()).replace(' ', '_'),
                                                              args.filtering_level, args.epochs)
    torch.save(PM, saved_as)
    print("\n Model successfully saved as {} ".format(saved_as))

    return PM


def apply_themes(args, device, model):
    img_path = os.path.join("photos", args.content_image)

    content_image = Image.open(img_path).resize(eval_size)

    masks = utils.get_masks(content_image, args.seg_threshold)
    filter_ids = utils.select_filters(masks, content_image, total_filters)

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_image = content_transform(content_image)
    content_images = content_image.expand(len(filter_ids), -1, -1, -1).to(device)
    # one forward pass to render themes
    with torch.no_grad():
        if args.load_model is None or args.load_model == "None":
            theme_model = model
        else:
            # our saved models were trained with gpu 3
            #theme_model = torch.load(args.load_model, map_location={'cuda:3': 'cuda:0'})
            theme_model = torch.load(args.load_model)
            theme_model.eval()
        theme_model.to(device)
        output = theme_model(content_images, filter_ids).cpu()

    output_list = []
    for img in output:
        img = img.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        output_list.append(img)

    rendered_themes = utils.render_themes(output_list, masks)
    utils.draw(rendered_themes, args.content_image, args.output_image)

def gram_matrix(x):
    (b, d, h, w) = x.size()
    features = x.view(b, d, w * h)
    gram = features.bmm(features.transpose(1, 2)) / (d * h * w)
    return gram


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--epochs", type=int, default=2,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="batch size for training")
    parser.add_argument("--gpu", type=int, default=3,
                        help="GPU id. -1 for CPU")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="learning rate")
    parser.add_argument("--save", type=str, default="saved_models",
                        help="path to folder where trained models will be saved.")
    parser.add_argument("--output-image", type=str, default="output",
                        help="path for saving the output image")
    parser.add_argument("--content-image", type=str, default="hoofer.jpg",
                        help="name of content image you want to apply_themes")
    parser.add_argument("--load-model", type=str,
                        default="saved_models/Sat_Dec__5_12:41:33_2020_filter_level_2.6_epoch2.pth",
                        help="saved model to be used for stylizing the image if applicable")
    parser.add_argument("--filtering-level", type=float, default=2.6,
                        help="A positive integer for degree of filtering.0 for no filter.")
    parser.add_argument("--seg_threshold", type=float, default=0.9,
                        help="Threshold for instance segmentation (between 0 and 1).")

    args = parser.parse_args()

    if args.gpu > -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        print("Start running on GPU{}".format(args.gpu))
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Running on CPU only.")

    if args.load_model is None or args.load_model == 'None':
        print("Start training from scratch")
        model = train(args, device)
        model.eval()
        apply_themes(args, device, model)
    else:
        print("Loading pretrained model from {}".format(args.load_model))
        apply_themes(args, device, None)


