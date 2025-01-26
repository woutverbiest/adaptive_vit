
from mmseg.datasets.builder import PIPELINES

# Register only if it hasn't been registered before
if 'CenterCrop' not in PIPELINES:

    @PIPELINES.register_module()
    class CenterCrop(object):
        """Center crop the image & seg.
        Args:
            crop_size (tuple): Expected size after cropping, (h, w).
        """
    
        def __init__(self, crop_size, ignore_index=255):
            assert crop_size[0] > 0 and crop_size[1] > 0
            self.crop_size = crop_size
            self.ignore_index = ignore_index
    
        def get_crop_bbox(self, img):
            """Randomly get a crop bounding box."""
            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = margin_h // 2
            offset_w = margin_w // 2
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
    
            return crop_y1, crop_y2, crop_x1, crop_x2
    
        def crop(self, img, crop_bbox):
            """Crop from ``img``"""
            crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            return img
    
        def __call__(self, results):
            """Call function to randomly crop images, semantic segmentation maps.
    
            Args:
                results (dict): Result dict from loading pipeline.
    
            Returns:
                dict: Randomly cropped results, 'img_shape' key in result dict is
                    updated according to crop size.
            """
    
            img = results['img']
            crop_bbox = self.get_crop_bbox(img)
    
            # crop the image
            img = self.crop(img, crop_bbox)
            img_shape = img.shape
            results['img'] = img
            results['img_shape'] = img_shape
    
            # crop semantic seg
            for key in results.get('seg_fields', []):
                results[key] = self.crop(results[key], crop_bbox)
    
            return results
    
        def __repr__(self):
            return self.__class__.__name__ + f'(crop_size={self.crop_size})'
