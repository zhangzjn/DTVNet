from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NPY_EXTENSIONS = [
    '.npy',
]

def is_npy_file(filename):
    return any(filename.endswith(extension) for extension in NPY_EXTENSIONS)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# data loader
def pil_loader(path):
    return Image.open(path).convert('RGB')


def default_loader(path):
    return pil_loader(path)
