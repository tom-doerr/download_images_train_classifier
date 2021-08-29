#!/usr/bin/env python3

'''
Adds the correct file ending to the files in the subfolders of 
the images directory. In case the correct file ending can not be 
determined, add '.jpg' as the file ending.
'''

import os

def main():
    '''
    The main function
    '''
    images_dir = 'images/'
    for item in os.listdir(images_dir):
        subdir = item
        if os.path.isdir(os.path.join(images_dir,subdir)):
            for file in os.listdir(os.path.join(images_dir,subdir)):
                if file.endswith('.jpg'):
                    continue
                elif file.endswith('.png'):
                    os.rename(os.path.join(images_dir,subdir,file),\
                              os.path.join(images_dir,subdir,file[:-4]+'.jpg'))
                # else:
                    # os.rename(os.path.join(images_dir,subdir,file[:-4]+'.png'),\
                              # os.path.join(images_dir,subdir,file[:-4]+'.jpg'))
                else:
                    os.rename(os.path.join(images_dir,subdir,file),\
                              os.path.join(images_dir,subdir,file[:-4]+'.jpg'))

if __name__ == '__main__':
    main()
