use ndarray::{Array2};
use std::fs::{OpenOptions, File};
use std::io::{BufWriter, Read, Write};
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

    pub fn is_lvl_critical(&self, x: usize, y: usize, critival_lvl: f32) -> (bool, bool) {
        (self.organics[(y,x)] > critival_lvl, self.electric[(y,x)] > critival_lvl)
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

    pub fn load(save_path: &Path) -> Result<Self, Box<dyn Error>> {
        // meta.txt: "height,width"
        let meta_path = save_path.join("meta.txt");
        if !meta_path.exists() {
            return Err(format!("meta.txt not found in {:?}", save_path).into());
        }
        let mut f = File::open(&meta_path)?;
        let mut contents = String::new();
        f.read_to_string(&mut contents)?;
        let first_line = contents.lines().next().ok_or("meta.txt is empty")?;
        let parts: Vec<&str> = first_line.trim().split(',').collect();
        if parts.len() < 2 {
            return Err("unexpected meta.txt format".into());
        }
        let height: usize = parts[0].parse()?;
        let width: usize = parts[1].parse()?;

        let organic_path = save_path.join("organic.npy");
        let electric_path = save_path.join("electric.npy");
        if !organic_path.exists() || !electric_path.exists() {
            return Err("organic.npy or electric.npy missing".into());
        }

        let organics: Array2<f32> = load_npy(&organic_path)?;
        let electric: Array2<f32> = load_npy(&electric_path)?;

        Ok(Map {
            height,
            width,
            organics,
            electric,
        })
    }
}