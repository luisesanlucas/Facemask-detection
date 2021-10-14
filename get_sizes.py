from utils import *

results, labels = get_images_and_labels(faces=True)
print(largest_image(results))