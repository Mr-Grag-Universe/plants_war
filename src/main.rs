use crate::common::Coord;
use crate::map::{Map};
use crate::simulation::{Simulation};
use crate::cells::*;
use rand::{rng, Rng};

use pbr::ProgressBar;

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
    let cells: Vec<Cell> = (0..n).into_par_iter().map_init(
        || rng(), // инициализация RNG для каждого потока; предположим rng() cheap
        |local_rng, _i| {
            let genome = Genome::random(1 + 2 * 7 * 7, 512, 512, 4 * 4, 0.0, 0.1);
            let cell = Cell {
                kind: CellKind::Storage(Storage { energy: 0.5, genome }),
                life_time: 100,
                pos: Coord {
                    x: local_rng.random_range(0..w) as i64,
                    y: local_rng.random_range(0..h) as i64,
                },
                out_dir: common::Direction::East,
                energy: 0.5,
            };
            cell
        },
    ).collect();

    cells
}


fn main() {
    let world_map = Map::new(1024, 1024);
    let mut simulation = Simulation::new(Some(world_map), String::from("saves"), String::from("snap"));
    println!("world generation...");
    simulation.add_cells(generate_cells_parallel(1024, 1024, 10));
    
    println!("\nrunning the world!");
    let n_runs = 100;
    let mut pb = ProgressBar::new(n_runs);
    for _ in 0..n_runs {
        simulation.step();
        if simulation.save_state().is_err() {
            println!("We broke around the saving of the state to file!");
            panic!("save error!");
        }
        simulation.save_iter += 1;
        pb.inc();
    }
    pb.finish_println("done");

    println!("Hello, world!");
}
