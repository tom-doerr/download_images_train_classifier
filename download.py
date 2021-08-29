#!/usr/bin/env python3

'''
Download images of certain class from the web.
The classes of images for which to download images are in the images_to_downloaded 
list. Save the images to disk.
'''

images_to_downloaded = ['cat', 'dog']

import os
import sys
import urllib.request
import argparse

import requests
from bs4 import BeautifulSoup
import urllib.parse

import logging
logger = logging.getLogger(__name__)


def download_images(images_to_downloaded, output_dir, num_images_to_download=100):
    '''
    Download the images of certain classes from the web.
    Args:
        images_to_downloaded: list of strings, the classes of images to download
        output_dir: string, path to the directory where to save the images
        num_images_to_download: int, number of images to download for each class
    '''
    for class_name in images_to_downloaded:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

        search_url = "https://www.google.co.in/search?q="+class_name+"&source=lnms&tbm=isch"
        header = {'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
        url = 'https://www.google.com/search?q=' + urllib.parse.quote(class_name) + '&source=lnms&tbm=isch'
        soup = BeautifulSoup(urllib.request.urlopen(urllib.request.Request(search_url, headers = header)),'html.parser')
        ActualImages = []
        for a in soup.find_all("img"):
            json_data = a.json_data
            link = a["src"]
            # image_type = json_data["ity"]
            image_type = link.split('.')[-1]
            ActualImages.append((link, image_type))

        for i, (img, Type) in enumerate(ActualImages[0:num_images_to_download]):
            try:
                link = img
                req = urllib.request.Request(link, headers=header)
                raw_img = urllib.request.urlopen(req).read()
                f = open(os.path.join(class_dir, os.path.basename(link)), "wb")
                f.write(urllib.request.urlopen(req).read())
                f.close()
                print("Image downloaded to ", class_dir)
            except Exception as e:
                print("could not load : "+img)
                print(e)

def main():
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-to-download', type=str, nargs='+',
                        help='list of classes of images to download')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='path to the directory where to save the images')
    parser.add_argument('--images_to_downloaded', type=str, nargs='+', required=True,
                        help='list of classes of images to download')
    parser.add_argument('--num_images_to_download', type=int, default=100,
                        help='number of images to download for each class')
    args = parser.parse_args()
    args = parser.parse_args()
    download_images(args.images_to_downloaded, args.output_dir, args.num_images_to_download)

if __name__ == "__main__":
    main()
