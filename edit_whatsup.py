import os

for img_path in os.listdir('whatsup/controlled_images'):
    if 'left' not in img_path and 'right' not in img_path:
        # delete the image
        os.remove(os.path.join('whatsup/controlled_images', img_path))

print("Cleanup done.")