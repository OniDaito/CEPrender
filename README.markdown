# swiss_parse

A couple of programs for dealing with Christian's data. This is written in rust and requires a few external libraries. This project creates two programs. One renders our [fits]() images with some augmentation. The other allows us to preview each image and say if we want to include it in the set or not.

## Requirements

* HDF5 libraries for the hdf5-rust package 
*[https://github.com/aldanor/hdf5-rust](https://github.com/aldanor/hdf5-rust)

Use git to checkout this project *at the same level* as  this one. E.g

    /home/me/projects/swiss_parse
    /home/me/projects/hdf5-rust

## Building

If you have rust installed, enter the swiss_parse directory and type

    cargo build

I think it's a good idea to make sure your rust compilier is up-to-date. I've fallen afoul of this a few times

    rustup upgrade

## Running

There are two programs: render and chooser. Render creates all the images and chooser lets you browse the images to select these that you want to use.

    cargo run --bin render --release <path to matlab file> <path to output> <threads> <sigma> <pertubations> <accepted - OPTIONAL>
    cargo run --bin chooser --release <path to matlab file>
