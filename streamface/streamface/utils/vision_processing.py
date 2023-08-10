import cv2


def detect_emptyness(image):
    """Emptyness detection based on edges

    Uses Canny edge detection

    Args:
        image (ndarray): Image as numpy array

    Returns:
        float: Fraction of pixels containing no edges
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(image, 0, 25).ravel()

    zeros = (edges == 0).nonzero()[0]

    zero_fraction = len(zeros) / len(edges)

    return zero_fraction


def detect_blur(image, image_size=(1280, 720), filter_size=11, center_crop=0.15):
    """Blur detection using Variance of Laplacian

    Args:
        image (ndarray): Image as numpy array
        image_size (2-tuple): Size the image will be resized to, None = No resizing
        filter_size (int): Size of square laplacian filter, default = 11
        center_crop (float): Fraction of image to discard from all sides, None = No cropping

    Returns:
        float: Clarity score (Low = Blurry)
    """
    if image_size is not None:
        image = cv2.resize(image, image_size, interpolation=cv2.INTER_NEAREST)

    if center_crop is not None:
        pH = int(center_crop * image.shape[0])
        pW = int(center_crop * image.shape[1])
        image = image[pH : -pH, pW : -pW]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    score = cv2.Laplacian(image, cv2.CV_64F, filter_size).var()

    return score


def template_matching(image, template, size=(1280, 720)):
    """Template matching using cv2.matchTemplate

    Metric = Normalized Squared Difference

    Args:
        image (ndarray): Test image as numpy array
        template (ndarray): Template image as numpy array
        size (2-tuple): Size the image will be resized to, None = No resizing

    Returns:
        float: Similarity score [0, 1] (Higher = Good match)
    """
    if size is not None:
        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        template = cv2.resize(template, size, interpolation=cv2.INTER_NEAREST)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    dst = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)[0, 0]

    sim = 1 - dst

    return sim
