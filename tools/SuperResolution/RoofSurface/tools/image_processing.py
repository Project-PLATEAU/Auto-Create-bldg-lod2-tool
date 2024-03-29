import os
import cv2


def save_images(images, output_folder, filename):
    """
    Save a list of images to the output folder.

    Parameters:
    - images: List of images to be saved.
    - output_folder: Path to the output folder.
    - filename: Original filename.

    Returns:
    - split_list: List of paths to the saved images.
    """
    split_folder_path = os.path.join(output_folder, 'split_images')
    os.makedirs(split_folder_path, exist_ok=True)

    split_list = []
    basename, ext = os.path.splitext(filename)
    for i, img in enumerate(images):
        output_filename = f"{basename}#{i}{ext}"
        output_path = os.path.join(split_folder_path, output_filename)
        split_list.append(output_path)
        cv2.imwrite(output_path, img)

    return split_list


def resized_image(image, gsd, gr=0.25):
    """
    Resize image to 0.25 [m] ground resolution. (downsampling only)

    Parameters:
    - image: input image.
    - gsd: Original ground resolution [m].
    - gr: Modified ground resolution [m]. Default is 0.25 [m].

    Returns:
    - resize_image: Resized image.
    """

    scale = gsd / gr

    height = int(image.shape[0] * scale)
    width = int(image.shape[1] * scale)

    resize_image = cv2.resize(image, (width, height), cv2.INTER_LINEAR)

    return resize_image


def split_image(input_path, gsd, size=120):
    """
    Split an image into smaller segments.

    Parameters:
    - input_path: Path to the input image.
    - gsd: Ground resolution [m].
    - size: Size of each segment.

    Returns:
    - (height, width, channel): Original image dimensions.
    - images: List of segmented images.
    """
    
    image = cv2.imread(input_path)

    if gsd < 0.25:
        image = resized_image(image, gsd)

    height, width, channel = image.shape
    images = []

    for y in range(0, height, size):
        for x in range(0, width, size):
            if x + size <= width and y + size <= height:
                images.append(image[y:y+size, x:x+size])
            else:
                over_x = min(x, width-size)
                over_y = min(y, height-size)
                images.append(image[over_y:over_y+size, over_x:over_x+size])

    return (height, width, channel), images


def merge_images(merged_image, image, num, size=480):
    """
    Merge a segmented image into a larger image.

    Parameters:
    - merged_image: The larger image to merge into.
    - image: The segmented image to merge.
    - num: Index of the segmented image.
    - size: Size of each segment.

    Returns:
    - merged_image: The updated merged image.
    """
    height, width, _ = merged_image.shape

    if width % size == 0:
        row = (num // (width // size)) * size
        col = (num % (width // size)) * size
    else:
        row = (num // ((width // size) + 1)) * size
        col = (num % ((width // size) + 1)) * size

    if row + size > height or col + size > width:
        row = min(row, height-size)
        col = min(col, width-size)

    merged_image[row:row + size, col:col + size] = image

    return merged_image
