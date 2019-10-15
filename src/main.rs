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
use std::process;
use scoped_threadpool::Pool;

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
    
    if args.len() < 5 {
        println!("Usage: swiss parse <path to matlab file> <threads> <sigma> <pertubations>"); 
        process::exit(1);
    }
    
    let nthreads = args[2].parse::<u32>().unwrap();
    let npertubations = args[4].parse::<u32>().unwrap();
    let sigma = args[3].parse::<f32>().unwrap();

    let models = parse_matlab(&args[1]);
    println!("Model 0 0 X: {},  Y: {}", models[0][0].x, models[0][0].y);
}
