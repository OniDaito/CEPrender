""" parse_christian.py - take Christian's matlab .mat 
files and parse them so we can extract the coordinates
and render a FITS image we can use in our network.

"""

import scipy.io
import h5py
import numpy as np
import math
import argparse
import random
from tqdm import tqdm
from astropy.io import fits

def render_coords(models, outdir, sigma, img_size = (128, 128), num_aug=1) :
    print("Number of Models:", len(models))
    print("Labels in model 0:", len(models[0]))
    print(models[0][0])
    idy = 0

    for idx in tqdm(range(len(models))):
        # scale the model for rendering
        model = models[idx]
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
        for _ in range(num_aug):
            rendered = [] 
            for _ in range(img_size[0]):
                iy = []
                for _ in range(img_size[1]):
                    iy.append(0.0)
                rendered.append(iy)

            # A random rotation around the plane
            rr = random.uniform(-math.pi, math.pi)
            rm = ((math.cos(rr), -math.sin(rr)), (math.sin(rr), math.cos(rr)))

            for ex in range(img_size[0]):
                for ey in range(img_size[1]):
                    for point in model_scaled:
                        # Perform a 2D rotation to augment
                        # TODO - could potentially do this in final image space?
                        xs = point[0] * rm[0][0] + point[1] * rm[0][1]
                        ys = point[0] * rm[1][0] + point[1] * rm[1][1]
                        xs += (img_size[0] / 2.0)
                        ys += (img_size[1] / 2.0)
                        if xs >= 0 and xs < img_size[0] \
                            and ys >= 0 and ys < img_size[1] :   
                            pval = (1.0 / (2.0 * math.pi * sigma**2)) * math.exp(-(((ex - xs)**2 + (ey - ys)**2) / (2*sigma**2)))        
                            rendered[ex][ey] += pval
                        else:
                            print("Point still exceeding range in image", idx)

            # Now write out a fits image
            fits_name = "chr_" + str(idy).zfill(5) + ".fits"
            hdu = fits.PrimaryHDU(np.array(rendered, dtype=np.float32))
            hdul = fits.HDUList([hdu])  
            hdul.writeto(outdir + "/" + fits_name)
            idy += 1

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Parse a mat file to generate real data')
    parser.add_argument('--file', 
        help='Path to the mat file')
    parser.add_argument('--outdir', default="./", 
        help='Directory to save output files')
    parser.add_argument('--sigma', type=float, default=1.2,\
        help='Sigma to render at.')  
    parser.add_argument('--numaug', type=int, default=10,\
        help='Number of random augmentations')  

    models = []
    args = parser.parse_args()
    try:
        mat = scipy.io.loadmat(args.file)
        print(mat)
    except NotImplementedError :
        with h5py.File(args.file, 'r') as f:
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
                                if name == b"/#refs#/K":
                                    print((data[0][idx], data[1][idx]))
                                #print("x, y", row[0], row[1])
                            models.append(coords)
                          
    #render_coords(models, args.outdir, args.sigma, num_aug = args.numaug)