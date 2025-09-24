use ndarray::{Array2, s};

pub struct Map {
    pub width: usize,
    pub height: usize,
    pub organics: Array2<f64>,
    pub electric: Array2<f64>,
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

    pub fn set_organics(&mut self, x: usize, y: usize, val: f64) {
        self.organics[(y,x)] = val;
    }
    pub fn set_electric(&mut self, x: usize, y: usize, val: f64) {
        self.electric[(y,x)] = val;
    }
}