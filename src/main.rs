use crate::common::Coord;
use crate::map::{Map};
use crate::simulation::{Simulation};
use crate::cells::*;

use std::path::Path;
use rand::{rng, Rng};
use pbr::ProgressBar;
use std::collections::HashSet;

use rayon::prelude::*;

pub mod map;
pub mod cells;
pub mod common;
pub mod simulation;


const N_RUNS: u64 = 3000;
const SAVE_INTERVAL: u64 = N_RUNS / 10;

const DEFAULT_N_CELLS: usize = 5000;
const DEFAULT_MAP_H: usize = 1024;
const DEFAULT_MAP_W: usize = 1024;
const DEFAULT_LIFETIME: i16 = 150;


fn generate_cells_parallel(h: usize, w: usize, n: usize) -> Vec<Cell> {
    // generate candidates in parallel
    let mut candidates: Vec<Cell> = (0..n).into_par_iter().map_init(
        || rng(),
        |local_rng, _i| {
            let genome = Genome::random(1 + 2 * 5 * 5, 128, 256, 4 * 4, 0.0, 0.1);
            Cell {
                kind: CellKind::Storage(Storage { genome }),
                life_time: DEFAULT_LIFETIME,
                pos: Coord {
                    x: local_rng.random_range(0..w) as i64,
                    y: local_rng.random_range(0..h) as i64,
                },
                out_dir: common::Direction::East,
                energy: 1.0,
            }
        },
    ).collect();

    // keep only first cell per unique coordinate (single-threaded, cheap)
    let mut seen = HashSet::with_capacity(candidates.len());
    let mut unique: Vec<Cell> = Vec::with_capacity(candidates.len());

    for cell in candidates.drain(..) {
        let key = (cell.pos.x, cell.pos.y);
        if seen.insert(key) {
            unique.push(cell);
        }
    }

    unique
}


fn create_new_simulation() -> Simulation {
    let world_map = Map::new(DEFAULT_MAP_W, DEFAULT_MAP_H);
    let mut s = Simulation::new(Some(world_map), 
                                                    String::from("saves"), 
                                                    String::from("snap"),
                                                    DEFAULT_LIFETIME);
    println!("world generation...");
    s.add_cells(generate_cells_parallel(DEFAULT_MAP_H, DEFAULT_MAP_W, DEFAULT_N_CELLS));
    s
}


fn main() {
    let mut simulation = Simulation::load(Path::new("./saves_back")).unwrap_or_else(|_| {
        println!("cannot load simulation");
        create_new_simulation()
    });
    
    println!("\nrunning the world!");
    let mut pb = ProgressBar::new(N_RUNS);
    for i in 0..N_RUNS {
        simulation.step();
        if simulation.save_view(false).is_err() {
            println!("We broke around the saving of the view to file!");
            panic!("save error!");
        }
        if i > 0 && i % SAVE_INTERVAL == 0 && simulation.save_state(true).is_err() {
            println!("We broke around the saving of the state to file!");
            panic!("save error!");
        }
        simulation.save_iter += 1;
        pb.inc();
    }
    pb.finish_println("done");

    println!("Hello, world!");
}
