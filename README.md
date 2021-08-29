This repo contains code that downloads images and trains an image classifier on them.
The `./train.py` function might still contain some bugs.
Almost all of the code was written by my [vim codex](https://github.com/tom-doerr/vim_codex) plugin.

Example usage:
```
./download.py  --output_dir images --images_to_download cat 
./download.py  --output_dir images --images_to_download dog 
./rename_images.py
./train.py
```
