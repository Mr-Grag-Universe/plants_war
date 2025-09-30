use ndarray::{Array2};
use std::fs::OpenOptions;
use std::io::{Write, BufWriter};
use std::path::Path;
use std::error::Error;
use crate::common::*;

pub struct Map {
    pub width: usize,
    pub height: usize,
    pub organics: Array2<f32>,
    pub electric: Array2<f32>,
}

impl Map {
    pub fn new(width: usize, height: usize) -> Self {
        let organics = Array2::zeros((height, width));
        let electric = Array2::zeros((height, width));
        Self { width, height, organics, electric }
    }

    pub fn in_bounds(&self, x: i64, y: i64) -> bool {
        x >= 0 && x < self.width as i64 && y >= 0 && y < self.height as i64
    }

    pub fn set_organics(&mut self, x: usize, y: usize, val: f32) {
        self.organics[(y,x)] = val;
    }
    pub fn set_electric(&mut self, x: usize, y: usize, val: f32) {
        self.electric[(y,x)] = val;
    }

    pub fn save(&self, save_path: &Path, overwrite: bool) -> Result<(), Box<dyn Error>> {
        let meta_path = save_path.join("meta.txt");
        if !meta_path.as_path().exists() || overwrite {
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(meta_path.as_path())?;
            let mut w = BufWriter::new(f);
            writeln!(w, "{},{}", self.height, self.width)?;
            w.flush()?;
        }

        let npy_path = save_path.join("organic.npy");
        save_npy(&self.organics, npy_path.as_path())?;

        let npy_path = save_path.join("electric.npy");
        save_npy(&self.electric, npy_path.as_path())?;

        Ok(())
    }
}