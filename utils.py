"""multi-filters and instance segmentation"""
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torchvision
import torch
import numpy as np
import cv2
import random
import sys
import time
import os

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()


def render_themes(filtered_imgs, masks):
    masks = np.stack((masks,) * 3, axis=-1)

    render_list = []
    for i in range(len(filtered_imgs)):
        mask = masks[i]
        filter = filtered_imgs[i][0:mask.shape[0], :, :]
        template = np.multiply(mask, filter)
        template = template.astype(np.uint8)
        render_list.append(template)
    return render_list


def color_masks(image):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = [0, 255, 0]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_masks(img, threshold):
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img).to(device)
    pred = model([img])

    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    masks = masks[:pred_t + 1]
    frame = np.zeros((masks.shape[1], masks.shape[2]))
    for mask in masks:
        frame[mask == 1] = 1
    bg = 1 - frame
    bg = np.array([bg])
    masks = np.append(masks, bg, axis=0)
    return masks


def select_filters(masks, img, num_filters):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    filter_ids = []

    for i in range(len(masks)):
        rgb_mask = color_masks(masks[i])
        temp_img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        plt.figure(figsize=(10,18))
        plt.imshow(temp_img)
        plt.show()
        # in case img show delayed
        #time.sleep(2)
        # prompt filter selection
        while True:
            if i < len(masks) -1:
                num = input("Please enter a filter id (0 to {0}) for person/obj {1} (-1 to quit):".format(num_filters-1, i))
            else:
                num = input("Please enter a filter id (0 to {0}) for the background (-1 to quit):".format(num_filters-1))
            try:
                id = int(num)
                if num_filters - 1 >= id >=0:
                    print("filter {0} selected".format(id, i))
                    filter_ids.append(id)
                    break
                elif id == -1:
                    print("Program terminated.")
                    sys.exit(0)
                else:
                    print("Error. filter id is not valid.")
            except ValueError:
                    print("Error. filter id should be an integer.")
    print("filter selection completed.")

    #return [0, 1, 4, 3, 7]
    return filter_ids


def draw(rendered_themes, img_name, output_path):
    print(rendered_themes[0].shape)
    canvas = np.zeros_like(rendered_themes[0]).astype(np.uint8)

    for obj in rendered_themes:
        canvas = cv2.add(canvas, obj)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(os.path.join(output_path, "filtered_{}".format(img_name)), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    plt.figure(figsize=(10, 18))
    plt.imshow(canvas)
    plt.show()






