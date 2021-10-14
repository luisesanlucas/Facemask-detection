"""
crop_faces.py

Created: 04/22/21 By: Sam
Modified: 04/30/21 By:Luis

This file digests XML files from the mask image dataset and uses them to locate and
extract human faces from images. It then saves the faces as separate images into 
separate folders according to their label. It runs on relative paths, so should be located
 in the same folder as images and annotation folders.
"""

# Built-in python packages
import os
import xml.etree.ElementTree as ET
# Python Imaging Library - "pip install --upgrade Pillow"
from PIL import Image


def crop_faces():
    """This just crops faces out of the dataset from a filesystem and saves them as
    separate images. Not strictly necessary, but enables direct feeding of faces
    into a NN."""
    xml_path = './archive/annotations/'
    img_path = './archive/images/'
    xml_files = list(sorted(os.listdir(xml_path)))
    for file in xml_files:
        name = file[:-4]
        img = Image.open(img_path + name + '.png')
        tree = ET.parse(xml_path + file)
        root = tree.getroot()
        i = 0
        for object in root.iter('object'):
            # Crop needs left, top, right, bottom (clockwise), box has two corners
            # box[0] - top left x val, box[1] - top left y val, others opposite
            crop = img.crop(
                (int(object[5][0].text), int(object[5][1].text), int(object[5][2].text), int(object[5][3].text))
            )
            print(crop)

            if object[0].text=='without_mask':
                crop.save('./images/no_mask/' + name + '-' + str(i) + '.png', format='png')
            elif object[0].text=='with_mask':
                crop.save('./images/mask/' + name + '-' + str(i) + '.png', format='png')
            elif object[0].text=='mask_weared_incorrect':
                crop.save('./images/incorrect_mask/' + name + '-' + str(i) + '.png', format='png')

            i += 1



crop_faces()
