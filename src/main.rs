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


// fn generate_cells(h: usize, w: usize, n: usize) -> Vec<Cell> {
//     let mut cells: Vec<Cell> = Vec::with_capacity(n);
//     let mut rng = rng(); // ваш генератор
//     let mut pb = ProgressBar::new(n as u64);

//     for _ in 0..n {
//         let genome = Genome::random(1 + 2 * 7 * 7, 512, 512, 4 * 4, 0.0, 0.1);
//         let cell = Cell {
//             kind: CellKind::Storage(Storage { energy: 0.5, genome }),
//             life_time: 100,
//             pos: Coord {
//                 x: rng.random_range(0..w) as i64,
//                 y: rng.random_range(0..h) as i64,
//             },
//             out_dir: common::Direction::East,
//             energy: 0.5,
//         };
//         cells.push(cell);
//         pb.inc();
//     }

//     pb.finish_println("done");
//     cells
// }

fn generate_cells_parallel(h: usize, w: usize, n: usize) -> Vec<Cell> {
    // generate candidates in parallel
    let mut candidates: Vec<Cell> = (0..n).into_par_iter().map_init(
        || rng(),
        |local_rng, _i| {
            let genome = Genome::random(1 + 2 * 5 * 5, 128, 256, 4 * 4, 0.0, 0.1);
            Cell {
                kind: CellKind::Storage(Storage { genome }),
                life_time: 50,
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


fn main() {
    // let world_map = Map::new(1024, 1024);
    // let mut simulation = Simulation::new(Some(world_map), 
    //                                                  String::from("saves"), 
    //                                                  String::from("snap"),
    //                                                  150);
    println!("world generation...");
    // simulation.add_cells(generate_cells_parallel(1024, 1024, 5000));
    let mut simulation = Simulation::load(Path::new("./saves_back")).expect("cannot load simulation");
    
    println!("\nrunning the world!");
    let n_runs = 10000;
    let mut pb = ProgressBar::new(n_runs);
    for i in 0..n_runs {
        simulation.step();
        if simulation.save_view(false).is_err() {
            println!("We broke around the saving of the view to file!");
            panic!("save error!");
        }
        if i > 0 && i % 100 == 0 && simulation.save_state(true).is_err() {
            println!("We broke around the saving of the state to file!");
            panic!("save error!");
        }
        simulation.save_iter += 1;
        pb.inc();
    }
    pb.finish_println("done");

    println!("Hello, world!");
}
