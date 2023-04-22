import os.path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.exposure
from copy import copy

def main():
    ##################### SET PARAMETERS HERE #####################
    # path to the depth images, depth images should be named like 0.png, 1.png, ..
    path_depth = "/home/tobias/Documents/GR/data/record_5fps_1080p_WFOV_2X2BINNED/depth"
    # path to the place where to write the labeles in the folder labels
    # optionally folders mask, overlay and colorized_depth are created and the respective data is written there
    path_write = "/home/tobias/Documents/GR/data"

    # set if depth overlay and colorized depth image should be created
    create_mask = True
    create_overlay = True
    create_colorize_depth = True

    # define range of depth images that will be labeled, [start_depth, end_depth)
    start_depth = 30
    end_depth = 125

    # define range of empty scene images, that will be averaged, [start_empty_images, end_empty_images)
    start_empty_images = 0
    end_empty_images = 16

    # different parameters for the labeling process
    num_persons_in_scene = 1
    num_max_masks = 10
    num_anchors = 10

    kernel_fine = np.ones((15, 15), np.uint8)
    kernel_anchor = np.ones((40, 40), np.uint8)
    kernel_post = np.ones((25, 25), np.uint8)

    distance_threshold_fine = 300
    distance_threshold_anchor = 1000
    ###############################################################

    # create empty mask to store average result
    depth_0 = cv2.imread(os.path.join(path_depth, "0.png"))
    img_height_width = (depth_0.shape[0], depth_0.shape[1])
    empty_img_avg = np.zeros(img_height_width)
    for i in range(start_empty_images, end_empty_images):
        empty_img_avg += cv2.imread(os.path.join(path_depth, "{}.png".format(i)), cv2.IMREAD_ANYDEPTH)
    empty_img_avg = empty_img_avg / (end_empty_images-start_empty_images)

    # start labeling process
    for img_num in range(start_depth, end_depth):
        img_person = cv2.imread(os.path.join(path_depth, "{}.png".format(img_num)), cv2.IMREAD_ANYDEPTH)

        mask_fine = np.zeros(img_height_width)
        mask_anchor = np.zeros(img_height_width)

        # extraction
        for i in range(mask_fine.shape[0]):
            for j in range(mask_fine.shape[1]):
                if abs(empty_img_avg[i, j]-img_person[i, j]) > distance_threshold_fine  and img_person[i, j] > 0:
                    mask_fine[i, j] = 1
                if abs(empty_img_avg[i, j]-img_person[i, j]) > distance_threshold_anchor  and img_person[i, j] > 0:
                    mask_anchor[i, j] = 1

        # opening and closing
        mask_open = cv2.morphologyEx(mask_fine, cv2.MORPH_OPEN, kernel_fine)
        mask_open_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_fine)
        mask_anchor = cv2.morphologyEx(mask_anchor, cv2.MORPH_OPEN, kernel_anchor)

        # find contours in fine mask and anchor mask
        mask_fine_uint8 = mask_open_close.astype(np.uint8)
        contours_fine, _ = cv2.findContours(mask_fine_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_anchor_uint8 = mask_anchor.astype(np.uint8)
        contours_anchor, _ = cv2.findContours(mask_anchor_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours after area size
        contours_fine_dict = {}
        for c in contours_fine:
            area = cv2.contourArea(c)
            contours_fine_dict[area] = c
        contours_fine_dict = dict(sorted(contours_fine_dict.items()))
        contours_fine_list = []

        contours_anchors_dict = {}
        for c in contours_anchor:
            area = cv2.contourArea(c)
            contours_anchors_dict[area] = c
        contours_anchors_dict = dict(sorted(contours_anchors_dict.items()))
        contour_anchors_list = []

        # only consider a specific maximal number of masks / contours to reduce complexity
        for i in range(min(num_max_masks, len(contours_fine_dict))):
            contours_fine_list.append(contours_fine_dict.popitem()[1])
        for i in range(min(num_anchors, len(contours_anchors_dict))):
            contour_anchors_list.append(contours_anchors_dict.popitem()[1])

        # draw the masks of the selected contours
        mask_selected_contours = np.zeros(mask_fine.shape)
        for c in contours_fine_list:
            cv2.drawContours(mask_selected_contours, [c], 0, 255, thickness=cv2.FILLED)

        mask_selected_anchors = np.zeros(mask_fine.shape)
        for c in contour_anchors_list:
            cv2.drawContours(mask_selected_anchors, [c], 0, 255, thickness=cv2.FILLED)

        # only keep the fine masks that have a non-zero intersection with at least one anchor
        contours_filtered = filter_masks(contours_fine_list, contour_anchors_list)

        # draw the final masks
        mask_filtered = np.zeros(img_height_width)
        for c in contours_filtered:
            cv2.drawContours(mask_filtered, [c], 0, 255, thickness=cv2.FILLED)

        # post-processing of the final mask to close potential holes
        mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, kernel_post)
        mask_filtered_uint8 = mask_filtered.astype(np.uint8)

        # calculate contours again, to find potentially fused masks
        contours_fine, _ = cv2.findContours(mask_filtered_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort the filtered contours
        contours_filtered_dict = {}
        for c in contours_fine:
            area = cv2.contourArea(c)
            contours_filtered_dict[area] = c
        contours_filtered_dict = dict(sorted(contours_filtered_dict.items()))

        # assumption that if specific number of persons are in the scene, the largest masks are from these persons
        labels = []
        contours_final = []
        for i in range(min(num_persons_in_scene, len(contours_filtered_dict))):
            contour = contours_filtered_dict.popitem()[1]
            contours_final.append(contour)
            label_line = '0 ' + ' '.join([f'{int(point[0][0]) / img_height_width[1]} {int(point[0][1]) / img_height_width[0]}' for point in contour]) + "\n"
            labels.append(label_line)

        # draw the final masks out of the final contours
        mask_final = np.zeros(img_height_width)
        for c in contours_final:
            cv2.drawContours(mask_final, [c], 0, 255, thickness=cv2.FILLED)

        # check if directory exist, if not create one
        path_labels = os.path.join(path_write, "labels")
        path_masks = os.path.join(path_write, "mask")
        path_overlay = os.path.join(path_write, "overlay")
        path_depth_colorized = os.path.join(path_write, "depth_colorized")

        if not os.path.exists(path_labels):
            os.mkdir(str(path_labels))
        if not os.path.exists(path_masks) and create_mask:
            os.mkdir(str(path_masks))
        if not os.path.exists(path_overlay) and create_overlay:
            os.mkdir(str(path_overlay))
        if not os.path.exists(path_depth_colorized) and create_colorize_depth:
            os.mkdir(str(path_depth_colorized))

        # colorize depth if wanted
        if create_overlay or colorize_depth_images:
            depth_color = colorize_depth(img_person)
            img_overlay = copy(depth_color)
            # create color mask
            for i in range(img_height_width[0]):
                for j in range(img_height_width[1]):
                    if mask_final[i, j] > 0:
                        img_overlay[i, j, 0] = 0
                        img_overlay[i, j, 1] = 0
                        img_overlay[i, j, 2] = 255

        # save labels and masks
        f = open(os.path.join(path_labels, "{}.txt".format(img_num)), "w")
        f.writelines(labels)
        f.close()

        if create_mask:
            cv2.imwrite(os.path.join(path_masks, "{}.jpg".format(img_num)), mask_final)   # binary mask
        if create_overlay:
            cv2.imwrite(os.path.join(path_overlay, "{}.jpg".format(img_num)), img_overlay)  # mask in color image
        if create_colorize_depth:
            cv2.imwrite(os.path.join(path_depth_colorized, "{}.jpg".format(img_num)), depth_color)  # mask in color image

        print("Labeled image number ", img_num, " with ", len(labels), " objects")

        # optional plot of the process, uncomment if needed
        """
        fig = plt.figure(figsize=(8, 8))
        columns = 1
        rows = 7
        ax1 = fig.add_subplot(rows, columns, 1)
        ax1.set_title("extraction")
        plt.imshow(mask_fine)
        ax2 = fig.add_subplot(rows, columns, 2)
        ax2.set_title("opening")
        plt.imshow(mask_open)
        ax3 = fig.add_subplot(rows, columns, 3)
        ax3.set_title("closing")
        plt.imshow(mask_open_close)
        ax4 = fig.add_subplot(rows, columns, 4)
        ax4.set_title("fine masks")
        plt.imshow(mask_selected_contours)
        ax5 = fig.add_subplot(rows, columns, 5)
        ax5.set_title("anchor masks")
        plt.imshow(mask_selected_anchors)
        ax6 = fig.add_subplot(rows, columns, 6)
        ax6.set_title("union")
        plt.imshow(mask_selected_contours+mask_selected_anchors)
        ax7 = fig.add_subplot(rows, columns, 7)
        ax7.set_title("final masks")
        plt.imshow(mask_filtered)
        fig.tight_layout(pad=0.5)
        plt.show()
        """


def filter_masks(contours, contours_anchor):
    # function that takes fine and anchor masks and keep a fine mask only if
    # it is a subset of at least one anchor mask
    contours_to_remain = []

    # create mask with anchors
    mask_anchors = np.zeros((1080, 1920))
    for c in contours_anchor:
        cv2.drawContours(mask_anchors, [c], 0, 255, thickness=cv2.FILLED)

    for c in contours:
        # get mask of individual contours
        mask_contour = np.zeros((1080, 1920))
        cv2.drawContours(mask_contour, [c], 0, 255, thickness=cv2.FILLED)
        non_zero_pixels = np.nonzero(mask_contour)
        x_list = non_zero_pixels[0]
        y_list = non_zero_pixels[1]

        for i in range(len(x_list)):
            x = x_list[i]
            y = y_list[i]
            # check if pixel lies on an anchor
            if mask_anchors[x, y] > 0:
                contours_to_remain.append(c)
                break

    return contours_to_remain


def colorize_depth(depth):
    # stretch to full dynamic range
    stretch = skimage.exposure.rescale_intensity(depth, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # convert to 3 channels
    stretch = cv2.merge([stretch, stretch, stretch])

    # define colors
    color1 = (0, 0, 255)  # red
    color2 = (0, 165, 255)  # orange
    color3 = (0, 255, 255)  # yellow
    color4 = (255, 255, 0)  # cyan
    color4_5 = (255, 128, 0)
    color5 = (255, 0, 0)  # blue
    color5_6 = (165, 32, 0)
    color6 = (128, 64, 64)  # violet

    # adjust for different color palette
    colorArr = np.array([[color4, color4_5, color5, color5_6, color6]], dtype=np.uint8)

    # resize lut to 256 (or more) values
    lut = cv2.resize(colorArr, (256, 1), interpolation=cv2.INTER_LINEAR)

    # apply lut
    result = cv2.LUT(stretch, lut)
    return result


if __name__ == '__main__':
    main()