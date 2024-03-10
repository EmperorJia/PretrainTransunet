import numpy as np
import scipy.ndimage as ndimage


# 五折交叉验证数据集划分
def cross_validation_5_split(label_0_list, label_1_list, label_2_list):
    split_list = []
    for i in range(5):
        split_list.append([])

    np.random.seed(0)

    np.random.shuffle(label_0_list)
    np.random.shuffle(label_1_list)
    np.random.shuffle(label_2_list)

    for i in range(5):
        split_list[i].extend(
            label_0_list[
                int(len(label_0_list) * 0.2 * i) : int(
                    len(label_0_list) * 0.2 * (i + 1)
                )
            ]
        )
        split_list[i].extend(
            label_1_list[
                int(len(label_1_list) * 0.2 * i) : int(
                    len(label_1_list) * 0.2 * (i + 1)
                )
            ]
        )
        split_list[i].extend(
            label_2_list[
                int(len(label_2_list) * 0.2 * i) : int(
                    len(label_2_list) * 0.2 * (i + 1)
                )
            ]
        )
        np.random.shuffle(split_list[i])

    return split_list


# 把图片的大小缩放到size
def np_resample_to_size(np_image, size, is_mask=False):
    original_size = np_image.shape
    resample_ratio = [
        size[0] / original_size[0],
        size[1] / original_size[1],
        size[2] / original_size[2],
    ]
    if is_mask:
        np_image = ndimage.zoom(np_image, resample_ratio, order=0)
    else:
        np_image = ndimage.zoom(
            np_image, [1, resample_ratio[1], resample_ratio[2]], order=1
        )
        np_image = ndimage.zoom(np_image, [resample_ratio[0], 1, 1], order=0)

    return np_image


# 把图片的大小缩放到size,2d版本
def np_resample_to_size_2d(np_image, size, is_mask=False):
    original_size = np_image.shape
    resample_ratio = [
        1,
        size[0] / original_size[1],
        size[1] / original_size[2],
    ]
    if is_mask:
        np_image = ndimage.zoom(np_image, resample_ratio, order=0)
    else:
        np_image = ndimage.zoom(np_image, resample_ratio, order=1)

    return np_image


# 从图片中提取mask并缩放到指定尺寸
def np_crop_according_to_mask(
    np_image, np_mask, original_mask_size, resample_size, xy_extend, z_extend
):

    z_min, z_max, y_min, y_max, x_min, x_max = np_get_bbox(np_mask)
    diff = (x_max - x_min) - (y_max - y_min)
    if diff > 0:
        y_max = y_max + diff / 2
        y_min = y_min - diff / 2
    else:
        x_max = x_max - diff / 2
        x_min = x_min + diff / 2

    z_min = int(max(0, z_min - z_extend))
    z_max = int(min(original_mask_size[0], z_max + z_extend))
    y_min = int(max(0, y_min - xy_extend))
    y_max = int(min(original_mask_size[1], y_max + xy_extend))
    x_min = int(max(0, x_min - xy_extend))
    x_max = int(min(original_mask_size[2], x_max + xy_extend))

    resample_ratio = [
        resample_size[0] / (z_max + 1 - z_min),
        resample_size[1] / (y_max + 1 - y_min),
        resample_size[2] / (x_max + 1 - x_min),
    ]

    np_image = np_image[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
    np_mask = np_mask[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]

    #order越大越平滑，但为什么要两步做插值？
    np_image = ndimage.zoom(
        np_image, [1, resample_ratio[1], resample_ratio[2]], order=1
    )
    #罪魁祸首：这里np_image(16,512,512),放缩因子[1.8823529411764706, 0.31189083820662766, 0.31189083820662766]，乘完约等于(30,160,160)
    np_image = ndimage.zoom(np_image, [resample_ratio[0], 1, 1], order=0)
    np_mask = ndimage.zoom(np_mask, resample_ratio, order=0)
    return np_image, np_mask, (z_min, z_max, y_min, y_max, x_min, x_max)

def reverse_resample(processed_image, bbox):
    """
    将处理后的图像逆向重采样到与原始裁剪图像相同的尺寸。
    
    参数:
    - processed_image: 经过处理并需要被放回原始位置的图像
    - original_bbox_size: 裁剪并重采样前的边界框尺寸，格式为(z, y, x)
    - bbox: 裁剪区域的边界框坐标，格式为(z_min, z_max, y_min, y_max, x_min, x_max)
    - original_mask_size: 原始掩码的大小，格式为(z, y, x)
    
    返回:
    - 逆向重采样后的图像
    """
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    
    # 计算逆向重采样的比率
    resample_ratio = [
        (z_max + 1 - z_min) / processed_image.shape[0],
        (y_max + 1 - y_min) / processed_image.shape[1],
        (x_max + 1 - x_min) / processed_image.shape[2],
    ]
    
    # 逆向重采样
    #【重要通知】这里进行0阶插值，因为mask的值非0或1，可以验证一下
    resampled_image = ndimage.zoom(processed_image, resample_ratio, order=0)
    
    return resampled_image

def insert_resampled_image_back(original_image, processed_image, bbox):
    """
    将经过处理和逆向重采样的图像放回到原始图像中对应的裁剪位置。
    
    参数:
    - original_image: 原始图像
    - processed_image: 经过处理的图像
    - bbox: 裁剪时使用的边界框坐标，格式为(z_min, z_max, y_min, y_max, x_min, x_max)
    - original_mask_size: 原始掩码的大小，格式为(z, y, x)
    - resample_size: 处理图像之前的重采样尺寸，用于逆向重采样计算
    
    返回:
    - 更新后的原始图像
    """
    
    # 逆向重采样处理后的图像
    resampled_image = reverse_resample(processed_image, bbox)
    
    # 将逆向重采样后的图像放回原位
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    updated_image = original_image.copy()
    updated_image[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = resampled_image
    
    return updated_image


# 从图片中提取mask并缩放到指定尺寸,2d版本
def np_crop_according_to_mask_2d(
    np_image, np_mask, original_mask_size, resample_size, xy_extend
):
    z_min, z_max, y_min, y_max, x_min, x_max = np_get_bbox(np_mask)
    diff = (x_max - x_min) - (y_max - y_min)
    if diff > 0:
        y_max = y_max + diff / 2
        y_min = y_min - diff / 2
    else:
        x_max = x_max - diff / 2
        x_min = x_min + diff / 2

    y_min = int(max(0, y_min - xy_extend))
    y_max = int(min(original_mask_size[1], y_max + xy_extend))
    x_min = int(max(0, x_min - xy_extend))
    x_max = int(min(original_mask_size[2], x_max + xy_extend))

    resample_ratio = [
        1,
        resample_size[1] / (y_max + 1 - y_min),
        resample_size[2] / (x_max + 1 - x_min),
    ]

    np_image = np_image[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
    np_mask = np_mask[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]

    np_image = ndimage.zoom(
        np_image, [1, resample_ratio[1], resample_ratio[2]], order=1
    )
    np_mask = ndimage.zoom(np_mask, resample_ratio, order=0)
    return np_image, np_mask, (z_min, z_max, y_min, y_max, x_min, x_max)


# 把图像的mask替换为另一张图片的mask
def np_change_mask(old_image, old_mask, new_image, new_mask):
    old_z_min, old_z_max, old_y_min, old_y_max, old_x_min, old_x_max = np_get_bbox(
        old_mask
    )
    new_z_min, new_z_max, new_y_min, new_y_max, new_x_min, new_x_max = np_get_bbox(
        new_mask
    )
    resample_ratio = [
        (new_z_max + 1 - new_z_min) / (old_z_max + 1 - old_z_min),
        (new_y_max + 1 - new_y_min) / (old_y_max + 1 - old_y_min),
        (new_x_max + 1 - new_x_min) / (old_x_max + 1 - old_x_min),
    ]
    c_image = old_image[
        old_z_min : old_z_max + 1, old_y_min : old_y_max + 1, old_x_min : old_x_max + 1
    ]
    c_mask = old_mask[
        old_z_min : old_z_max + 1, old_y_min : old_y_max + 1, old_x_min : old_x_max + 1
    ]
    c_image = ndimage.zoom(c_image, [1, resample_ratio[1], resample_ratio[2]], order=1)
    c_image = ndimage.zoom(c_image, [resample_ratio[0], 1, 1], order=0)
    c_mask = ndimage.zoom(c_mask, resample_ratio, order=0)
    new_image[
        new_z_min : new_z_max + 1, new_y_min : new_y_max + 1, new_x_min : new_x_max + 1
    ] = c_image
    new_mask[
        new_z_min : new_z_max + 1, new_y_min : new_y_max + 1, new_x_min : new_x_max + 1
    ] = c_mask

    return new_image, new_mask


# 得到边界框
def np_get_bbox(np_mask):
    z_sum = np.sum(np_mask, axis=(1, 2))
    y_sum = np.sum(np_mask, axis=(0, 2))
    x_sum = np.sum(np_mask, axis=(0, 1))

    z_min = np.nonzero(z_sum)[0][0]
    z_max = np.nonzero(z_sum)[0][-1]
    y_min = np.nonzero(y_sum)[0][0]
    y_max = np.nonzero(y_sum)[0][-1]
    x_min = np.nonzero(x_sum)[0][0]
    x_max = np.nonzero(x_sum)[0][-1]

    return (z_min, z_max, y_min, y_max, x_min, x_max)


# 恢复到mask原尺寸
def np_reshape(image_size, np_mask, bbox):
    new_mask = np.zeros(image_size)
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    np_mask_size = np_mask.shape
    resample_ratio = [
        (z_max + 1 - z_min) / np_mask_size[0],
        (y_max + 1 - y_min) / np_mask_size[1],
        (x_max + 1 - x_min) / np_mask_size[2],
    ]
    np_mask = ndimage.zoom(np_mask, resample_ratio, order=0)
    new_mask[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] = np_mask
    return new_mask


# 灰度裁剪
def np_clip_window(np_image):
    np_image = np.clip(np_image, 0, 5000)
    return np_image


# 归一化

def np_normalize(np_image):
    np_image = (np_image) / np.max(np_image)
    return np_image


# 旋转
def np_rotate(np_image, np_mask, angle):
    if angle != 0:
        np_image = ndimage.rotate(np_image, angle, axes=(1, 2), reshape=False, order=1)
        np_mask = ndimage.rotate(np_mask, angle, axes=(1, 2), reshape=False, order=0)
    return np_image, np_mask


# 翻转
def np_flip(np_image, np_mask, axis):
    if axis != -1:
        np_image = np.flip(np_image, axis)
        np_mask = np.flip(np_mask, axis)
    return np_image, np_mask


# 随机旋转
def np_random_rotate(np_image, np_mask, angle_list):
    angle = np.random.choice(angle_list)
    if angle != 0:
        np_image, np_mask = np_rotate(np_image, np_mask, angle)
    return np_image, np_mask


# 随机翻转
def np_random_flip(np_image, np_mask, axis_list):
    axis = np.random.choice(axis_list)
    if axis != -1:
        np_image, np_mask = np_flip(np_image, np_mask, axis)
    return np_image, np_mask


# 计算Dice
def np_dice(input, target):
    smooth = 1e-5
    iflat = input.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
