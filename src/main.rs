/// A small program that parses Christian's 
/// MATLAB data files that he sent us. Based
/// on the python version I wrote but it's 
/// much faster!
///
/// Author: Benjamin Blundell
/// Email: me@benjamin.computer

extern crate rand;
extern crate image;
extern crate nalgebra as na;
extern crate probability;
extern crate scoped_threadpool;
extern crate fitrs;
extern crate rand_distr;
extern crate hdf5;

use std::env;
use std::fmt;
use rand::prelude::*;
use fitrs::{Fits, Hdu};
use rand_distr::{Normal, Distribution};
use std::process;
use rand::distributions::Uniform;
use scoped_threadpool::Pool;
use std::sync::mpsc::channel;
use pbr::ProgressBar;

// This represents a row in the matlab file
// 0 and 1 are x and y but I can't remember what the others
// are so we just use placeholders for now.
// DITCHED for now as it won't convert for some reason.
/*#[derive(hdf5::H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct SwissRow {
    x : f32,
    y : f32,
    a : i32,
    b : f32,
    c : f32,
    d : f32,
    e : f32,
    f : f32,
    g : f32,
    h : f32,
    i : i32,
    j : i32,
    k : i32
}

impl fmt::Display for SwissRow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}*/

static WIDTH : u32 = 128;
static HEIGHT : u32 = 128;

pub struct Point {
    x : f32,
    y : f32
}

// Scale and move all the points so they are in WIDTH, HEIGHT
// and the Centre of mass moves to the origin.
fn scale_shift_model( model : &Vec<Point> ) -> Vec<Point> {
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
    let diag =((maxx - minx) * (maxx - minx) + (maxy - miny) * (maxy - miny)).sqrt();
    // Make scalar a little smaller after selecting the smallest
    let scalar = (WIDTH as f32 / diag).min(HEIGHT as f32 / diag) * 0.7;
        
     for point in model {
        let np = Point {
            x : (point.x - com.0) * scalar,
            y : (point.y - com.1) * scalar
        };
        scaled.push(np);
    } 
    scaled
}

pub fn save_fits(img : &Vec<Vec<f32>>, filename : &String) {
    let mut data : Vec<f32> = (0..HEIGHT)
        .map(|i| (0..WIDTH).map(
               move |j| (i + j) as f32)).flatten().collect();

    for _y in 0..HEIGHT {
        for _x in 0..WIDTH {
            let idx : usize = (_y * WIDTH +_x ) as usize; 
            data[idx] = img[_x as usize][(HEIGHT - _y - 1) as usize];
            // / intensity * MULTFAC;
        }
    }

    let mut primary_hdu = 
        Hdu::new(&[WIDTH as usize , HEIGHT as usize], data);
    // Insert values in header
    primary_hdu.insert("NORMALISATION", "NONE");
    primary_hdu.insert("WIDTH", WIDTH as i32);
    primary_hdu.insert("HEIGHT", HEIGHT as i32);
    Fits::create(filename, primary_hdu).expect("Failed to create");  
}


fn render (models : &Vec<Vec<Point>>, out_path : &String,  nthreads : u32, 
    pertubations : u32, sigma : f32) {
    // Split into threads here I think
    let pi = std::f32::consts::PI;
    let (tx, rx) = channel();
    let mut progress : i32 = 0;
    let mut pool = Pool::new(nthreads);

    let num_runs = models.len() as u32;
    let truns = (num_runs / nthreads) as u32;
    let spare = (num_runs % nthreads) as u32;
    let mut pb = ProgressBar::new(num_runs as u64);
    pb.format("╢▌▌░╟");

    pool.scoped(|scoped| {
        for _t in 0..nthreads {
            let tx = tx.clone();
            let start : usize = (_t * truns) as usize;
            let mut end = ((_t + 1)  * truns) as usize;
            if _t == nthreads - 1 { end = end + (spare as usize); }
            let cslice = &models[start..end];
           
            scoped.execute( move || { 
                let mut rng = thread_rng();
                let side = Uniform::new(-pi, pi);

                for _i in 0..cslice.len() {
                    let scaled = scale_shift_model(&cslice[_i]);
                
                    for _j in 0..pertubations {
                        let mut timg : Vec<Vec<f32>> = vec![];

                        // Could be faster I bet
                        for _x in 0..WIDTH {
                            let mut tt : Vec<f32> = vec![];
                            for _y in 0..HEIGHT { tt.push(0.0); }
                            timg.push(tt);
                        }
                        // A random rotation around the plane
                        let rr = rng.sample(side);
                        let rm = (rr.cos(), -rr.sin(), rr.sin(), rr.cos());

                        for ex in 0..WIDTH {
                            for ey in 0..HEIGHT {
                                for point in &scaled {
                                    let xs = point.x * rm.0 + point.y * rm.1;
                                    let ys = point.x * rm.2 + point.y * rm.3;
                                    let xf = xs + (WIDTH as f32/ 2.0);
                                    let yf = ys + (HEIGHT as f32 / 2.0);
                                    if xf >= 0.0 && xf < WIDTH as f32 && yf >= 0.0 && yf < HEIGHT as f32 {   
                                        let pval = (1.0 / (2.0 * pi * sigma.powf(2.0))) *
                                            (-((ex as f32 - xf).powf(2.0) + (ey as f32 - ys).powf(2.0)) / (2.0*sigma.powf(2.0))).exp();        
                                        timg[ex as usize][ey as usize] += pval;
                                    }
                                    else {
                                        // TODO - ideally we send an error that propagates
                                        // and kills all other threads and quits cleanly
                                        println!("Point still exceeding range in image");
                                    }
                                }
                            }
                        }
                        
                        let fidx = format!("/image_{:06}.fits",
                            (start + _i * (pertubations as usize))  + _j as usize);
                        let mut fitspath = out_path.clone();
                        fitspath.push_str(&fidx);
                        save_fits(&timg, &fitspath);
                    }
                    tx.send(_i).unwrap();
                }
            });
        }

        // Update our progress bar
        while progress < num_runs as i32 {
            match rx.try_recv() {
                Ok(_a) => {
                    pb.inc();
                    progress = progress + 1;
                }, Err(_e) => {}
            }
        }
    });
}

fn parse_matlab(path : &String) -> Vec<Vec<Point>> {
    let mut models : Vec<Vec<Point>>  = vec!();

    match hdf5::File::open(path) {
        Ok(file) => {
            for t in file.member_names() {
                for s in t {
                    println!("{}", s);
                }
            }

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
                                        }
                                    }
                                    models.push(model);
                                }, 
                                Err(e) => {
                                    println!("{}", e);
                                }
                            }
                        }
                    }
                },
                Err(e) => { println!("{}", e); }
            }

        }, Err(e) => {
            println!("Error opening file: {} {}", path, e);
        }
    }
    models
} 

fn main() {
     let args: Vec<_> = env::args().collect();
    
    if args.len() < 6 {
        println!("Usage: swiss parse <path to matlab file> <path to output> <threads> <sigma> <pertubations>"); 
        process::exit(1);
    }
    
    let nthreads = args[3].parse::<u32>().unwrap();
    let npertubations = args[5].parse::<u32>().unwrap();
    let sigma = args[4].parse::<f32>().unwrap();

    let models = parse_matlab(&args[1]);
    render(&models, &args[2], nthreads, npertubations, sigma);
}
