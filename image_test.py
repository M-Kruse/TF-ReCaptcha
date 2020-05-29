import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Create figure and axes
fig,ax = plt.subplots(1)

# Create a Rectangle patch
#start_coords, x_size, y_size

def open_image(img_path):
	return np.array(Image.open(img_path), dtype=np.uint8)

def draw_rect(np_img, size_tuple, x_dim, y_dim):
	rect = patches.Rectangle(size_tuple, x_dim, y_dim, linewidth=1, edgecolor='r', facecolor='none')
	# Add the patch to the Axes
	ax.add_patch(rect)

im = open_image('test_500x500.png')
ax.imshow(im)
draw_rect(im, (40, 100), 100, 100)
plt.show()