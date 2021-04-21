# CEPrender

A couple of programs for dealing with the [CEP152](https://www.ncbi.nlm.nih.gov/gene/22995) MATLAB/HDF5 data from [Suliana Manley's group](https://www.epfl.ch/labs/leb/). This program is written in [Rust](https://www.rust-lang.org/) and requires a few external libraries. This project creates two programs. One renders our [fits](https://fits.gsfc.nasa.gov/) images with some augmentation. The other allows us to preview each image and say if we want to include it in the set or not.

## Requirements

* Linux (may work on other systems - so far untested)
* HDF5 libraries for the hdf5-rust package 
* [https://github.com/aldanor/hdf5-rust](https://github.com/aldanor/hdf5-rust)

Use git to checkout this project *at the same level* as  this one. E.g

    /home/me/projects/CEPrender
    /home/me/projects/hdf5-rust

## Building

If you have rust installed, enter the swiss_parse directory and type

    cargo build

Cargo should find the hdf5-rust package and build it, so long as it's at the same level as this project.

I think it's a good idea to make sure your rust compilier is up-to-date. I've fallen afoul of this a few times

    rustup upgrade

## Running

There are two programs: render and chooser. Render creates all the images and chooser lets you browse the images to select these that you want to use.

    cargo run --bin render --release <path to matlab file> <path to output> <threads> <sigma> <pertubations> <accepted - OPTIONAL>
    cargo run --bin chooser --release <path to matlab file>

Example:

    cargo run --release --bin render /tmp/Cep152_all.mat /tmp/cep152/1.8 24 1.8 100 accepted.txt
