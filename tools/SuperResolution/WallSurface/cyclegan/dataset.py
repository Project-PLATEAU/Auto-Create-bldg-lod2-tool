import cv2

from .transform import ImageTransformer

class DatasetDataLoader:
    def __init__(self, cfg_cyclegan):
        self.direction = cfg_cyclegan['direction']
        self.cfg_dataset = cfg_cyclegan['dataset']
        self.cfg_dataset['input_nc'] = self.cfg_dataset['output_nc'] if self.direction == 'BtoA' else self.cfg_dataset['input_nc']

        self.transform_list = self.transform(grayscale=(self.cfg_dataset['input_nc'] == 1))
        self.transformer = ImageTransformer

    def transform(self, params=None, grayscale=False, method=cv2.INTER_CUBIC, convert=True):
        transform_list = []

        if grayscale:
            transform_list.append(self.transformer.mean)

        if 'resize' in self.cfg_dataset['preprocess']:
            transform_list.append(lambda img: self.transformer.resize(img, self.cfg_dataset['load_size'], self.cfg_dataset['load_size'], method))
        elif 'scale_width' in self.cfg_dataset['preprocess']:
            transform_list.append(lambda img: self.transformer.scale_width(img, self.cfg_dataset['load_size'], method))

        if 'crop' in self.cfg_dataset['preprocess']:
            if params is None:
                transform_list.append(lambda img: self.transformer.random_crop(img, self.cfg_dataset['load_size']))
            else:
                transform_list.append(lambda img: self.transformer.crop(img, params['crop_pos'], self.cfg_dataset['load_size']))

        if self.cfg_dataset['preprocess'] == 'none':
            transform_list.append(lambda img: self.transformer.make_power_2(img, base=4, method=method))

        if not self.cfg_dataset['no_flip']:
            if params is None:
                transform_list.append(lambda img: self.transformer.random_horizontal_flip(img))
            elif params['flip']:
                transform_list.append(lambda img: self.transformer.flip(img, params['flip']))

        if convert:
            transform_list.append(lambda img: self.transformer.normalize(img, 0.5, 0.5))

        return transform_list

    
    def read_img(self, img, img_path):
        height, width, _ = img.shape

        for trans in self.transform_list:
            img = trans(img)
        img = img.unsqueeze(dim=0)

        if self.direction == 'AtoB':
            return {'A': img, 'A_paths': img_path}
        else:
            return {'B': img, 'B_paths': img_path}
        