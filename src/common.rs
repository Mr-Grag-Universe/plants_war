use ndarray::{Array2};
use serde::de::value::Error;
use std::path::Path;
use ndarray_npy::{write_npy, WriteNpyError, ReadNpyError, ReadNpyExt};
use std::io;
use std::fs;
use std::fs::File;

#[derive(Debug)]
pub struct Coord { pub x: i64, pub y: i64 }
impl Coord {
    pub fn to_tuple_yx(&self) -> (i64, i64) {
        (self.y, self.x)
    }
    pub fn to_tuple_xy(&self) -> (i64, i64) {
        (self.x, self.y)
    }
    pub fn clone(&self) -> Coord {
        Coord { x: self.x, y: self.y }
    }
    
    pub fn shift_inplace(&mut self, dir: &Direction) {
        match dir {
            Direction::North => { self.y -= 1 },
            Direction::South => { self.y += 1 },
            Direction::East  => { self.x += 1 },
            Direction::West  => { self.x -= 1 },
        };
    }

    pub fn shift(&self, dir: &Direction) -> Coord {
        let mut coord = self.clone();
        coord.shift_inplace(dir);
        return coord;
    }
}

#[derive(Clone)]
#[derive(Debug)]
pub enum Direction { North, East, South, West }
impl Direction {
    pub fn from_str(s: &str) -> Result<Direction, Error> {
        match s {
            "North" => Ok(Direction::North),
            "South" => Ok(Direction::South),
            "West"  => Ok(Direction::West),
            "East"  => Ok(Direction::East),
            _ => panic!("wrong direction in str-dir conversion!")
        }
    }
    pub fn all_directions() -> [Direction; 4] {
        [Direction::North, Direction::East, Direction::South, Direction::West]
    }

    pub fn oposite(&self) -> Direction {
        match self {
            Direction::East => Direction::West,
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            Direction::West => Direction::East,
        }
    }
}

#[derive(Debug)]
pub enum ResourceType { Solar, Organic, Electricity }

impl ResourceType {
    pub fn from_str(s: &str) -> Result<Self, Error> {
        match s {
            "Solar"       => Ok(Self::Solar),
            "Electricity" => Ok(Self::Electricity),
            "Organic"     => Ok(Self::Organic),
            _ => panic!("there is not such resource type!")
        }
    }

    pub fn default() -> Self {
        return Self::Solar;
    }
}

pub struct Action(pub Direction, pub u8);

pub fn save_npy<T: ndarray_npy::WritableElement>(arr: &Array2<T>, path: &Path) -> Result<(), WriteNpyError> {
    write_npy(path, arr)
}

pub fn load_npy<T: ndarray_npy::ReadableElement>(path: &Path) -> Result<Array2<T>, ReadNpyError> {
    let f = File::open(path).map_err(|e| ReadNpyError::Io(e))?;
    Array2::read_npy(f)
}

pub fn ensure_dir(path: &Path) -> io::Result<()> {
    if path.exists() {
        if path.is_dir() {
            Ok(())
        } else {
            Err(io::Error::new(io::ErrorKind::AlreadyExists, "path exists and is not a directory"))
        }
    } else {
        fs::create_dir_all(path)
    }
}