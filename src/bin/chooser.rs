/// A small program that parses Christian's 
/// MATLAB data files that he sent us. It 
/// renderers and then asks us to choose if
/// we want to keep it.
///
/// Using a little gtk-rs
/// https://gtk-rs.org/docs-src/tutorial/
///
/// Author: Benjamin Blundell
/// Email: me@benjamin.computer

extern crate rand;
extern crate image;
extern crate nalgebra as na;
extern crate probability;
extern crate scoped_threadpool;
extern crate hdf5;
extern crate ndarray;
extern crate gtk;
extern crate gio;
extern crate gdk_pixbuf;
extern crate glib;

use gtk::prelude::*;
use gio::prelude::*;
use gdk_pixbuf::Pixbuf;
use gdk_pixbuf::Colorspace;
use glib::Bytes;
use glib::clone;

use std::env;
use std::fmt;
use std::fs::OpenOptions;
use std::io::prelude::*;
use rand::prelude::*;
use std::sync::{Arc, Mutex};
use std::{cell::Cell, rc::Rc};

use std::process;
use rand::distributions::Uniform;
use scoped_threadpool::Pool;
use std::sync::mpsc::channel;
use pbr::ProgressBar;
use ndarray::{Slice, SliceInfo, s, Array1};

use gtk::{Application, ApplicationWindow, Button};

static WIDTH : u32 = 128;
static HEIGHT : u32 = 128;
static SHRINK : f32 = 0.95;

// Our point structure for the final spots
pub struct Point {
    x : f32,
    y : f32
}

// Holds our models and our GTK+ application
pub struct Chooser {
    app: gtk::Application,
    models : Vec<Vec<Point>>,
    model_index : Cell<usize> // use this so we can mutate it later
}


// go through all the models and find the extents. This gives
// us a global scale, we can use in the rendering.
fn find_extents ( models : &Vec<Vec<Point>> ) -> (f32, f32) {
    let mut w : f32 = 0.0;
    let mut h : f32  = 0.0;

    for model in models {
        let mut minx : f32 = 1e10;
        let mut miny : f32 = 1e10;
        let mut maxx : f32 = -1e10;
        let mut maxy : f32 = -1e10;
        
        for point in model {
            if point.x < minx { minx = point.x; }
            if point.y < miny { miny = point.y; }
            if point.x > maxx { maxx = point.x; }
            if point.y > maxy { maxy = point.y; }
        }
        let tw = (maxx - minx).abs();
        let th = (maxy - miny).abs();
        if tw > w { w = tw; }
        if th > h { h = th; }
    }

    (w, h)
}


// Get some stats on the models, starting with the mean and
// median number of points

fn find_stats ( models : &Vec<Vec<Point>> ) -> (f32, u32, f32, u32, u32) {
    let mut mean : f32 = 0.0;
    let mut median : u32 = 0;
    let mut min : u32 = 100000000;
    let mut max : u32 = 0;
    let mut sd : f32 = 0.0;
    let mut vv : Vec<u32> = vec![];

    for model in models {
        let ll = model.len();
        vv.push(ll as u32);
        if (ll as u32) < min {
            min = ll as u32;
        } 
        if (ll as u32) > max {
            max = ll as u32;
        }
        mean = mean + model.len() as f32;
    }
    
    vv.sort();
    median = vv[ (vv.len() / 2) as usize] as u32;
    mean = mean / vv.len() as f32;
    let vlen = vv.len();

    for ll in vv {
        sd = (ll as f32 - mean) * (ll as f32 - mean);
    }
    sd = (sd / vlen as f32).sqrt();
    
    (mean, median, sd, min, max)
}


// Scale and move all the points so they are in WIDTH, HEIGHT
// and the Centre of mass moves to the origin.
// We pass in the global scale as we don't want to scale per image.
// We are moving the centre of mass to the centre of the image though
// so we have to put in translation to our final model
fn scale_shift_model( model : &Vec<Point>, scale : f32 ) -> Vec<Point> {
    let mut scaled : Vec<Point> = vec![];
    let mut minx : f32 = 1e10;
    let mut miny : f32 = 1e10;
    let mut maxx : f32 = -1e10;
    let mut maxy : f32 = -1e10;

    for point in model {
        if point.x < minx { minx = point.x; }
        if point.y < miny { miny = point.y; }
        if point.x > maxx { maxx = point.x; }
        if point.y > maxy { maxy = point.y; }
    }

    let com = ((maxx + minx) / 2.0, (maxy + miny) / 2.0);
    /*let diag =((maxx - minx) * (maxx - minx) + (maxy - miny) * (maxy - miny)).sqrt();
    // Make scalar a little smaller after selecting the smallest
    let scalar = (WIDTH as f32 / diag).min(HEIGHT as f32 / diag) * SHRINK;*/
    let scalar = scale * (WIDTH as f32) * SHRINK;
        
     for point in model {
        let np = Point {
            x : (point.x - com.0) * scalar,
            y : (point.y - com.1) * scalar
        };
        scaled.push(np);
    } 
    scaled
}


// Render out our image to a block of memory
fn render (model : &Vec<Point>,  nthreads : u32, 
    sigma : f32, scale : f32) -> Bytes { 
    // Split into threads here I think
    let pi = std::f32::consts::PI;
    let mut progress : i32 = 0;
    let mut pool = Pool::new(nthreads);

    let num_runs = model.len() as u32;
    let truns = (num_runs / nthreads) as u32;
    let spare = (num_runs % nthreads) as u32;
    
    //let start : usize = (_t * truns) as usize;
    //let mut end = ((_t + 1)  * truns) as usize;
    //if _t == nthreads - 1 { end = end + (spare as usize); }

    let mut timg : Vec<Vec<f32>> = vec![];
    let scaled = scale_shift_model(&model, scale);
    
    // Could be faster I bet
    for _x in 0..WIDTH {
        let mut tt : Vec<f32> = vec![];
        for _y in 0..HEIGHT { tt.push(0.0); }
        timg.push(tt);
    }

    println!("Model size: {}", scaled.len());
    
    for ex in 0..WIDTH {
        for ey in 0..HEIGHT {
            for point in &scaled {
                let xs = point.x;
                let ys = point.y;
                let xf = xs + (WIDTH as f32/ 2.0);
                let yf = ys + (HEIGHT as f32 / 2.0);
                if xf >= 0.0 && xf < WIDTH as f32 && yf >= 0.0 && yf < HEIGHT as f32 {   
                    let pval = (1.0 / (2.0 * pi * sigma.powf(2.0))) *
                        (-((ex as f32 - xf).powf(2.0) + (ey as f32 - yf).powf(2.0)) / (2.0*sigma.powf(2.0))).exp();        
                    timg[ex as usize][ey as usize] += pval;
                }
                // We may get ones that exceed but it's very likely they are outliers
                /*else {
                    // TODO - ideally we send an error that propagates
                    // and kills all other threads and quits cleanly
                    println!("Point still exceeding range in image");
                }*/
            }
        }
    }        


    // Find min/max
    let mut min_v : f32 = 100000.0;
    let mut max_v : f32 = -100000.0;
    for ex in 0..WIDTH as usize {
        for ey in 0..HEIGHT as usize {
            if timg[ex][ey] < min_v { min_v = timg[ex][ey]; }
            if timg[ex][ey] > max_v { max_v = timg[ex][ey]; }
        }
    }
    
    // convert our float vec vec to an array of u8
    let mut bimg : Vec<u8> = vec![];
    for ex in 0..WIDTH as usize {
        for ey in 0..HEIGHT as usize {
            // GTK insists we have RGB so we triple everything :/
            for _ in 0..3 {
                bimg.push((timg[ex][ey] / max_v * 255.0) as u8);
            }
        }   
    }

    let b = Bytes::from(&bimg);
    b
}

// Parse the not quite proper MATLAB / HDF5 file passed in from the command line
fn parse_matlab(path : &String, models : &mut Vec<Vec<Point>>) -> Result<bool, hdf5::Error> {
    let mut file_option = 0;

    match hdf5::File::open(path) {
        Ok(file) => {
            for t in file.member_names() {
                for s in t {
                    println!("{}", s);
                    if s == "DBSCAN_filtered" {
                        file_option = 1;
                    }
                    if s == "Cep152_all_filtered" {
                        file_option = 2;
                    }
                }
            }
            // We have a different setup if we are using Christians all_Particles file
            if file_option == 1 {
                match file.dataset("DBSCAN_filtered") {
                    Ok(refs) => {
                        /*match refs.read_2d::<f32>() {
                            Ok(final_data) => {
                                let xpos = final_data.row(0);
                                let ypos = final_data.row(1);

                                println!("{}, {}", xpos, ypos);
                            
                            },
                            Err(e) => {
                                println!("Error in final data read. {}", e);
                            }
                        }*/
                        //let slice = s![1..-1, 1..-1];
                        //let slinfo = SliceInfo::<_, ndarray::Ix2>::new(slice).unwrap().as_ref();
                        //let slice = ndarray::SliceInfo::<usize, ndarray::Ix2>::new(0);
                        // No idea why an _ works for the second type :S
                        /*match refs.read_slice_1d::<f32, _>(s![2,..]) {
                            Ok(fdata) => {
                                println!("{:?}", fdata);
                            },
                            Err(e) => {
                                println!("Error reading slice {}", e);
                            }
                        }*/
                        println!("{:?}", refs.shape());
                        println!("{:?}", refs.chunks());
                        let v: Array1<f32> = refs.read_slice_1d(s![3,1..5])?;

                        
                        println!("Successful read.");
                        
                    },
                    Err(e) => {
                        return Err(e);
                    }
                }
            } else if file_option == 2 {
                // Cep152_all_filtered file
                 match file.group("#refs#") {
                    Ok(refs) => {
                        for names in refs.member_names() {
                            for name in names {
                                let mut owned_string: String = "/#refs#/".to_owned();
                                let borrowed_string: &str = &name;
                                owned_string.push_str(borrowed_string);

                                match file.dataset(&owned_string) {
                                    Ok(tset) => {
                                        //println!("{}", owned_string);
                                    
                                        match tset.read_2d::<f32>() {
                                            Ok(final_data) => {
                                                if final_data.shape()[0] == 8 {
                                                    let mut model : Vec<Point> = vec![];
                                                    let xpos = final_data.row(0);
                                                    let ypos = final_data.row(1);
                                                    //println!("{} {}", xpos, ypos);

                                                    for i in 0..tset.shape()[1] {
                                                        let p = Point {
                                                            x : xpos[i],
                                                            y : ypos[i]
                                                        };
                                                    
                                                        model.push(p);
                                                    }

                                                    models.push(model);   
                                                }
                                                //println!("{:?}", final_data.shape());
                                            },
                                            Err(e) => {
                                                println!("{}", e);
                                            }
                                        }
                                    },
                                    Err (e) => {
                                        println!("{}", e);
                                    }
                                }
                            }
                        }

                    },
                    Err(e) => {

                    }
                }
                
            } else {
                match file.group("#refs#") {
                    Ok(refs) => {
                        for names in refs.member_names() {
                            for name in names {
                                let mut owned_string: String = "/#refs#/".to_owned();
                                let borrowed_string: &str = &name;
                                owned_string.push_str(borrowed_string);

                                match file.dataset(&owned_string){
                                    Ok(tset) => {
                                        //println!("DataSet Shape: {:?}", tset.shape());
                                        let mut model : Vec<Point> = vec![];

                                        match tset.read_2d::<f32>() {
                                            Ok(final_data) => {
                                                let xpos = final_data.row(0);
                                                let ypos = final_data.row(1);

                                                for i in 0..tset.shape()[1] {
                                                    let p = Point {
                                                        x : xpos[i],
                                                        y : ypos[i]
                                                    };
                                                    
                                                    model.push(p);
                                                }
                                            },
                                            Err(e) => {
                                                println!("Error in final data read. {}", e);
                                                return Err(e);
                                            }
                                        }
                                        models.push(model);
                                    }, 
                                    Err(e) => {
                                        println!("{}", e);
                                        return Err(e);
                                    }
                                }
                            }
                        }
                    },
                    Err(e) => {
                        println!("{}", e); 
                        return Err(e);
                    }
                }
            }

        }, Err(e) => {
            println!("Error opening file: {} {}", path, e);
            return Err(e);
        }     
    }
    Ok(true)
} 

// Convert our model into a gtk::Image that we can present to
// the screen.
fn get_image(model : &Vec<Point>, scale : f32 ) -> gtk::Image {
    let first_image : Bytes = render(&model, 1, 1.25, scale);

    let pixybuf = Pixbuf::new_from_bytes(&first_image,
        Colorspace::Rgb,
        false, 
        8,
        WIDTH as i32,
        HEIGHT as i32,
        (WIDTH * 3 * 1) as i32
    );

    let image : gtk::Image = gtk::Image::new_from_pixbuf(Some(&pixybuf));
    image
}


// Our chooser struct/class implementation. Mostly just runs the GTK
// and keeps a hold on our models.
impl Chooser {
    pub fn new(path : &String) -> Rc<Self> {
        let app = Application::new(
            Some("com.github.gtk-rs.examples.basic"),
            Default::default(),
        ).expect("failed to initialize GTK application");

        let mut models : Vec<Vec<Point>> = vec!();
        let mut model_index : Cell<usize> = Cell::new(0);

        match parse_matlab(&path, &mut models) {
            Ok(_) => {
            }, 
            Err(e) => {
                println!("Error parsing MATLAB File: {}", e);
                process::exit(1);
            }
        }
        
        let chooser = Rc::new(Self {
            app,
            models,
            model_index
        });

        chooser
    }

    pub fn run(&self, app: Rc<Self>) {
        let app = app.clone();
        let args: Vec<String> = env::args().collect();

         // Find the stats on the models
         let (w, h) = find_extents(&self.models);
         let (mean, median, sd, min, max) = find_stats(&self.models);
         let cutoff = median - ((2.0 * sd) as u32);
 
         println!("Model sizes (min, max, mean, median, sd) : {}, {}, {}, {}, {}", 
             min, max, mean, median, sd);
         let mut scale = 3.0 / w;
         if h > w { scale = 3.0 / h; }
         println!("Scalar: {}, {}", scale, scale * (WIDTH as f32) * SHRINK); 
 
        self.app.connect_activate( move |gtkapp| {
            let window = ApplicationWindow::new(gtkapp);
            window.set_title("First GTK+ Program");
            window.set_default_size(350, 350);
            let vbox = gtk::Box::new(gtk::Orientation::Vertical, 3);
            let ibox = gtk::Box::new(gtk::Orientation::Horizontal, 1);
            let hbox = gtk::Box::new(gtk::Orientation::Horizontal, 3);
            let image = get_image(&(app.models[0]), scale);
            ibox.add(&image);
            vbox.add(&ibox);
            vbox.add(&hbox);
            window.add(&vbox);

            // Now look at buttons
            let button_accept = Button::new_with_label("Accept");
            let ibox_arc = Arc::new(Mutex::new(ibox));
            let ibox_accept = ibox_arc.clone();
            let mut app_accept = app.clone();

            let mut i : i32 = 0;
            let button_click = || { i + 1 };

            button_accept.connect_clicked( move |button| {
                println!("Accepted {}", app_accept.model_index.get());

                // Check we aren't overrunning
                let mi = app_accept.model_index.get();
                if mi >= app_accept.models.len() {
                    println!("All models checked!");
                    process::exit(1);
                }

                // Write out this index to a file saying it has been accepted
                let mut file = OpenOptions::new()
                    .write(true)
                    .append(true)
                    .open("accepted.txt")
                    .unwrap();

                if let Err(e) = writeln!(file, "{}", mi) {
                    eprintln!("Couldn't write to acceptance file: {}", e);
                }
                    
                let ibox_ref = ibox_accept.lock().unwrap();
                let children : Vec<gtk::Widget> = (*ibox_ref).get_children();
                app_accept.model_index.set(mi + 1);
                let image = get_image(&(app_accept.models[mi + 1]), scale);
                (*ibox_ref).remove(&children[0]);
                (*ibox_ref).add(&image);
                let window_ref = (*ibox_ref).get_parent().unwrap();
                window_ref.show_all();
            });

            hbox.add(&button_accept);

            let button_reject = Button::new_with_label("Reject");
            let ibox_reject = ibox_arc.clone();
            let mut app_reject = app.clone();

            button_reject.connect_clicked( move |button| {
                println!("Rejected {}", app_reject.model_index.get());

                // Check we aren't overrunning
                let mi = app_reject.model_index.get();
                if mi >= app_reject.models.len() {
                    println!("All models checked!");
                    process::exit(1);
                }

                let ibox_ref = ibox_reject.lock().unwrap();
                let children : Vec<gtk::Widget> = (*ibox_ref).get_children();
                app_reject.model_index.set(mi + 1);
                let image = get_image(&(app_reject.models[mi + 1]), scale);
                (*ibox_ref).remove(&children[0]);
                (*ibox_ref).add(&image);
                let window_ref = (*ibox_ref).get_parent().unwrap();
                window_ref.show_all();
            });

            hbox.add(&button_reject);
            window.show_all()

        });

        self.app.run(&[]);
    }
}

fn main() {
    let args: Vec<_> = env::args().collect();

    let mut models : Vec<Vec<Point>> = vec!();
    
    if args.len() < 2 {
        println!("Usage: swiss parse <path to matlab file>"); 
        process::exit(1);
    }

    gtk::init().expect("Unable to start GTK3");
    let app = Chooser::new(&args[1]);
    app.run(app.clone());
}