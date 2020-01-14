#!/usr/bin/env python3
from astropy.io import fits
import argparse
import os
import io
import tkinter as tk
from PIL import Image, ImageTk

WIDTH = HEIGHT = 128
accepted = []
rejected = []

class Application(tk.Frame):
    def __init__(self, filelist, master=None):
        super(Application, self).__init__(master)
        self.grid()
        assert(len(filelist) > 0)
        self.createWidgets(filelist[0])

    def convert_fits(self, fits_image):
        hdul = fits.open(fits_image)[0].data.byteswap().newbyteorder()
        image = Image.open(io.BytesIO(hdul))
        return image

    def createWidgets(self, first_path):
        photo = ImageTk.PhotoImage(self.convert_fits(first_path))
        self.mondialLabel = tk.Label(self, image=photo)
        self.mondialLabel.config(bg="#00ffff")
        self.mondialLabel.grid()
        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.quitButton.grid()
        
def create_file_list(path):
    final_list = []
    for dirname, dirnames, filenames in os.walk(path):
      for filename in filenames:
        img_extentions = ["fits", "FITS"]
        if any(x in filename for x in img_extentions):
            # We need to check there are no duffers in this list
            fpath = os.path.join(path, filename)
            final_list.append(fpath)
    
    return final_list

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='PyTorch dataload')
    parser.add_argument('--path', help='Path to the rendered files.')   
    args = parser.parse_args()
    assert(len(args.path) > 0)
    files_to_cx = create_file_list(args.path)
    app = Application(files_to_cx)
    app.master.title('Swiss Choice')
    app.mainloop()