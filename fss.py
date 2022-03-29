import cv2
from skimage.metrics import structural_similarity

if __name__ == "__main__":
    img_path = "cat.jpg"
    img = cv2.imread(img_path, 0)
    ref_img = img
    ssim = structural_similarity(ref_img, img)
    print(ssim)