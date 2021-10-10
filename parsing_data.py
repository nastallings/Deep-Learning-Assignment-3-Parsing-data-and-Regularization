import xml.etree.ElementTree as ET
import cv2
from zipfile import ZipFile
import os


def parse_XML_file(filelocation):
    tree = ET.parse(filelocation)
    root = tree.getroot()
    image_id = root.find('filename').text
    img = cv2.imread('Data/VOCdevkit/VOC2012/JPEGImages/' + image_id)

    for count, child in enumerate(root.findall('./object/')):
        # Get Object Title
        if child.tag == 'name':
            object = child.text
            if not os.path.isdir('Intermediate_Data_Set/' + object):
                os.mkdir('Intermediate_Data_Set/' + object)
        # Get Bounding boxes
        if child.tag == 'bndbox':
            bbox = [0, 0, 0, 0]
            for box in child:
                if box.tag == 'xmin':
                    bbox[0] = int(float(box.text))
                if box.tag == 'ymin':
                    bbox[1] = int(float(box.text))
                if box.tag == 'xmax':
                    bbox[2] = int(float(box.text))
                if box.tag == 'ymax':
                    bbox[3] = int(float(box.text))
            pic_of_object = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite('Intermediate_Data_Set/' + object + '/' + object + '_' + str(count) + '_' + image_id, cv2.resize(pic_of_object, (224, 224), interpolation = cv2.INTER_CUBIC))


if __name__ == "__main__":
    # Create new directory
    if not os.path.isdir('Intermediate_Data_Set'):
        os.mkdir('Intermediate_Data_Set')
    path = 'Data/VOCdevkit/VOC2012/Annotations/'
    for xmlFile in os.listdir(path):
        parse_XML_file(path + xmlFile)

    # Create Binary File
    zf = ZipFile("Intermediate_Data_Set.zip", "w")
    for dirname, subdirs, files in os.walk("Intermediate_Data_Set"):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()