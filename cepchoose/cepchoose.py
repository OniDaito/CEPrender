#!/usr/bin/env python3
from astropy.io import fits
import argparse
import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import scipy.io
import random
from threading import Thread
import h5py
import io
import math
import renderer
import tkinter as tk
from PIL import Image, ImageTk

WIDTH = HEIGHT = 128
accepted = [] # Indicies into the file in question
rejected = []
scratch_pad = []

class Application(tk.Frame):
    def __init__(self, models, master=None):
        super(Application, self).__init__(master)
        self.grid()
        assert(len(models) > 0)
        self.createWidgets(models[0])
    
    def convert_points(self, points):
        global scratch_pad
        render_coords(points, 1.25)
        nv = np.array(scratch_pad, dtype=np.float32)
        nv /= nv.max() * 255
        
        img = Image.fromarray(nv, mode="F")
        img = img.convert("L")
        return img

    def createWidgets(self, first_model):
        photo = ImageTk.PhotoImage(self.convert_points(first_model))
        self.imageLabel = tk.Label(self, image=photo)
        self.imageLabel.config(bg="#00ffff")
        self.imageLabel.grid()
        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.quitButton.grid()
        self.acceptButton = tk.Button(self, text='Accept', command=self.quit)
        self.acceptButton.grid()
        self.rejectButton = tk.Button(self, text='Reject', command=self.quit)
        self.rejectButton.grid()

class RenderThread (Thread):
    def __init__(self, threadID, points, start, end, img_size, sigma):
        Thread.__init__(self)
        self.threadID = threadID
        self.strt = start
        self.end = end
        self.sigma = sigma
        self.img_size = img_size
        self.points = points

    def run(self):
        global scratch_pad
        for ex in range(self.strt, self.end):
            for ey in range(self.img_size[1]):
                for point in self.points:
                    xs = point[0] + (self.img_size[0] / 2.0)
                    ys = point[1] + (self.img_size[1] / 2.0)
                    if xs >= 0 and xs < self.img_size[0] \
                        and ys >= 0 and ys < self.img_size[1] :   
                        pval = (1.0 / (2.0 * math.pi * self.sigma**2)) * math.exp(-(((ex - xs)**2 + (ey - ys)**2) / (2*self.sigma**2)))        
                        scratch_pad[ex][ey] += pval
                    else:
                        print("Point still exceeding range in image")

def render_coords(model, sigma, img_size = (128, 128)) :
    # scale the model for rendering
    model_scaled = []
    minx = 100000.0
    miny = minx
    maxx = -100000.0
    maxy = maxx
    
    for point in model:
        if point[0] > maxx:
            maxx = point[0]
        if point[0] < minx:
            minx = point[0]
        if point[1] > maxy:
            maxy = point[1]
        if point[1] < miny:
            miny = point[1]
    
    com = ((maxx + minx) / 2.0, (maxy + miny) / 2.0)
    diag = math.sqrt((maxx - minx) * (maxx - minx) + (maxy - miny) * (maxy - miny))
    scalar = min(img_size[0] / diag, img_size[1] / diag)
    
    # Make the scalar a bit bigger so we can include larger
    # sigma values and rotate a bit
    # TODO - is there an issue with noise and outliers? I mean
    # a single outlier could really shrink the images!
    scalar = scalar * 0.7

    # Move to the origin and scale
    for point in model:
        nx = (point[0] - com[0]) * scalar
        ny = (point[1] - com[1]) * scalar
        model_scaled.append((nx, ny))
    
    # Now augment our images a bit
    global scratch_pad
    scratch_pad = []
    for _ in range(img_size[0]):
        iy = []
        for _ in range(img_size[1]):
            iy.append(0.0)
        scratch_pad.append(iy)
    
    # This is far too taxing on our poor CPU
    nthreads = 4
    render_threads = []
  
    for i in range(nthreads):
        #try:
        start = int(math.floor(img_size[0] / nthreads * i))
        end = int(math.floor(img_size[0] / nthreads * (i + 1) - 1))
        render_thread = RenderThread(i, model_scaled, start, end, img_size, 1.25)
        print(type(render_thread))
        render_threads.append( render_thread )
        render_thread.start()
        
        #except Exception as e:
        #    print("Error: unable to start thread", e, type(e))

    for t in render_threads:
        t.join()

def parse_matlab(filepath):
    models = []
    try:
        mat = scipy.io.loadmat(filepath)
        print(mat)
    except NotImplementedError :
        with h5py.File(filepath, 'r') as f:
            #print("Keys", f.keys())
            for k, v in f.items():
                npv = np.array(v)
                #print (npv)
                if isinstance(npv[0], np.ndarray) :
                    #print("New Model")
                    for pref in npv[0]:
                        if isinstance(pref, h5py.Reference):
                            #print(npv[0][0].__class__)
                            name = h5py.h5r.get_name(pref, f.id)
                            #print(name)
                            data = f[name].value
                            coords = []
                            #print("-----")
                            #for i in range(len(data)) :
                            #    print(data[i][0])

                            for idx in range(len(data[0])) :
                                # 0 and 1 columns refer to x and y respectively.
                                coords.append((data[0][idx], data[1][idx]))
                                #if name == b"/#refs#/K":
                                #    print((data[0][idx], data[1][idx]))
                                #print("x, y", row[0], row[1])
                            print("New Model of size:", len(coords))
                            models.append(coords)
                          

    print("Number of models:", len(models))
    return models

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='PyTorch dataload')
    parser.add_argument('--path', help='Path to the Matlab file.')   
    args = parser.parse_args()
    assert(len(args.path) > 0)
    models = parse_matlab(args.path)
    print("Number of models:", len(models))
    #app = Application(models)
    #app.master.title('Swiss Choice')
    #app.mainloop()