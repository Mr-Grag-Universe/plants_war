use std::collections::{HashMap};
use rand::{prelude::*, rng};
use ndarray::{s, Array2, SliceInfo, Dim, SliceInfoElem};
use std::cmp;
use std::fs::OpenOptions;
use std::io::{Write, BufWriter};
use std::path::Path;

use crate::common::*;
use crate::map::{Map};
use crate::cells::*;


fn shuffled_indices(n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rand::rng();
    idx.shuffle(&mut rng);
    idx
}


pub struct Simulation {
    cells: HashMap<(i64,i64), Cell>,
    coords: Vec<Coord>,
    world_map: Map,

    
    pub save_iter: usize,
    save_path: String,
    save_file_name: String,
}

impl Simulation {
    pub fn new(world_map: Option<Map>, save_path: String, save_file_name: String) -> Self {
        let world_map: Map = world_map.unwrap_or_else(|| Map::new(1024, 1024));
        let cells: HashMap<(i64,i64), Cell> = HashMap::new();
        let coords: Vec<Coord> = Vec::new();
        let save_iter = 0;
        Simulation { 
            cells, 
            coords,
            world_map, 
            save_iter,
            save_path,
            save_file_name
        }
    }

    const WIN_W: usize = 7;
    const WIN_H: usize = 7;
    const PAD_VALUE: f32 = -1.0; // или 0.0

    fn extract_window(
        map: &Array2<f32>, // или другой тип элемента
        coord: &Coord,      // { x: usize, y: usize }
    ) -> Array2<f32> {
        // центральное окно размером WIN_H x WIN_W, с центром в coord
        // в вашем примере вы брали от x-3..x+4 (7 ширина) и y-3..y+4
        let left = coord.x as isize - 3;
        let top = coord.y as isize - 3;

        let mut out = Array2::from_elem((Self::WIN_H, Self::WIN_W), Self::PAD_VALUE);

        // вычисляем пересечение с исходной картой
        let map_h = map.nrows() as isize;
        let map_w = map.ncols() as isize;

        // диапазоны в координатах карты
        let src_x0 = cmp::max(left, 0) as usize;
        let src_y0 = cmp::max(top, 0) as usize;
        let src_x1 = cmp::min(left + Self::WIN_W as isize, map_w) as usize; // exclusive
        let src_y1 = cmp::min(top + Self::WIN_H as isize, map_h) as usize; // exclusive

        // куда копировать в выходной матрице
        let dst_x0 = (src_x0 as isize - left) as usize;
        let dst_y0 = (src_y0 as isize - top) as usize;
        let dst_x1 = dst_x0 + (src_x1 - src_x0);
        let dst_y1 = dst_y0 + (src_y1 - src_y0);

        if src_x0 < src_x1 && src_y0 < src_y1 {
            let src_view = map.slice(s![src_y0..src_y1, src_x0..src_x1]);
            out.slice_mut(s![dst_y0..dst_y1, dst_x0..dst_x1]).assign(&src_view);
        }

        out
    }

    pub fn add_cells(&mut self, cells: Vec<Cell>) {
        for cell in cells {
            self.coords.push(cell.pos.clone());
            self.cells.insert(cell.pos.to_tuple_xy(), cell);
        }
    }

    fn increase_polution(world_map: &mut Map, coord: &Coord) {
        let l_x = cmp::max(coord.x-1, 0) as usize;
        let t_y = cmp::max(coord.y-1, 0) as usize;
        let r_x = cmp::min(coord.x+2, world_map.width as i64) as usize;
        let b_y = cmp::min(coord.y+2, world_map.height as i64) as usize;
        let slice = s![t_y..b_y, l_x..r_x];
        let mut area = world_map.organics.slice_mut(slice);
        area += 0.1;
        let mut area = world_map.electric.slice_mut(slice);
        area += 0.1;
    }

    fn update_energy_dir(world_map: &Map, 
                        coord: &Coord, 
                        cell: &Cell, 
                        cells: &HashMap<(i64,i64), Cell>) -> (Coord, Option<Direction>) {
        let mut rec_coord = coord.shift(&cell.out_dir);
        let mut rec_dir: Option<Direction> = Some(cell.out_dir.clone());
        if !world_map.in_bounds(rec_coord.x, rec_coord.y) {
            panic!("Cannot transfer energy out of map bounds!");
        }
        if !cells.contains_key(&rec_coord.to_tuple_xy()) {
            // try to find another neighbour to transfer energy
            for dir in Direction::all_directions() {
                let rec_coord_tmp = coord.shift(&dir);
                if cells.contains_key(&rec_coord_tmp.to_tuple_xy()) {
                    let Some(cell) = cells.get(&rec_coord_tmp.to_tuple_xy()) else {
                        panic!("There is not such cell!");
                    };
                    match cell.kind {
                        CellKind::Producer(_) => { continue; },
                        _ => {
                            // new receiver found
                            rec_coord = rec_coord_tmp;
                            rec_dir = Some(dir.clone());
                            // new_dir.push(DirChange(rec_coord.clone(), dir, cell.dir_uuid));
                            break;
                        }
                    }
                }
            }
            if let Some(d) = &rec_dir { println!("contains!"); }
            else { println!("Did not find!"); }
        }
        (rec_coord.clone(), rec_dir)
    }

    fn get_slice(center_coord : &Coord, r: i64, h: usize, w: usize) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
        let l_x = cmp::max(center_coord.x-r, 0) as usize;
        let t_y = cmp::max(center_coord.y-r, 0) as usize;
        let r_x = cmp::min(center_coord.x+r+1, w as i64) as usize;
        let b_y = cmp::min(center_coord.y+r+1, h as i64) as usize;
        s![t_y..b_y, l_x..r_x]
    }

    pub fn step(&mut self) {
        let order = shuffled_indices(self.cells.len());
        
        let mut new_coords: Vec<Coord> = Vec::new();

        for i in order {
            let mut coord = self.coords[i].clone();
            let kind: &CellKind;
            {
                let Some(cell) = self.cells.get(&coord.to_tuple_xy()) else {
                    panic!("There is no cell with such coords {coord:?}!");
                };
                
                // if it's dead
                if cell.life_time <= 0 {
                    self.cells.remove(&coord.to_tuple_xy());
                    Self::increase_polution(&mut self.world_map, &coord);
                    continue;
                }

                kind = &cell.kind
            }

            match kind {
                CellKind::Producer(p) => {
                    // calculate direction to store energy
                    let mut rec_coord: Coord = Coord { x: 0, y: 0 };
                    let mut rec_dir: Option<Direction> = Some(Direction::North);
                    if let Some(cell) = self.cells.get(&coord.to_tuple_xy()) {
                        // calculate direction to store energy
                        (rec_coord, rec_dir) = Simulation::update_energy_dir(&self.world_map, &coord, cell, &self.cells);
                        if !self.cells.contains_key(&rec_coord.to_tuple_xy()) {
                            // kill cell (there is no receiver)
                            continue;
                        }
                    }
                    // println!("Producer cell: {:?} {:?}", rec_coord, rec_dir);

                    if !self.cells.contains_key(&rec_coord.to_tuple_xy()) {
                        // kill cell (there is no receiver)
                        continue;
                    }
                    // let rec_uuid = self.cells[&rec_coord.to_tuple()].dir_uuid;

                    // produced energy
                    let mut energy_produced = 0f32;
                    match p.resource {
                        ResourceType::Solar => {
                            energy_produced = 0.1f32;
                        },
                        ResourceType::Organic => {
                            let slice = Self::get_slice(&coord, 1, self.world_map.height, self.world_map.width);
                            let mut area =  self.world_map.organics.slice_mut(slice);
                            if area.sum() > 0.0f32 {
                                area -= 0.2;
                                area.map_inplace(|v| if *v < 0.0 { *v = 0.0 });
                                energy_produced = 0.1f32;
                            }
                        },
                        ResourceType::Electricity => {
                            let slice = Self::get_slice(&coord, 1, self.world_map.height, self.world_map.width);
                            let mut area =  self.world_map.electric.slice_mut(slice);
                            if area.sum() > 0.0f32 {
                                area -= 0.2;
                                area.map_inplace(|v| if *v < 0.0 { *v = 0.0 });
                                energy_produced = 0.1f32;
                            }
                        },
                    }
                    
                    { // energy increase
                        let rec_cell = self.cells.get_mut(&rec_coord.to_tuple_xy());
                        if let Some(c) = rec_cell {
                            c.energy += energy_produced;
                        } else {
                            panic!("There is not such cell! rec_coord is probably wrong here!");
                        }
                    }
                    { // actual direction for energy flow set
                        let Some(cell) = self.cells.get_mut(&coord.to_tuple_xy()) else {
                            panic!("There is no cell with such coords  {coord:?}!");
                        };
                        if let Some(d) = rec_dir {
                            cell.out_dir = d;
                        } else {
                            panic!("rec_dir is None for some reason here  {coord:?}");
                        }
                    }
                    // // add energy to a storage
                    // self.storages
                    //     .entry(rec_uuid)
                    //     .and_modify(|s| s.energy += energy_produced)
                    //     .or_insert_with(|| Storage { energy: energy_produced });
                },
                CellKind::Conductor => {
                    let mut rec_coord: Coord = Coord { x: 0, y: 0 };
                    let mut rec_dir: Option<Direction> = Some(Direction::North);
                    let mut energy = 0f32;
                    if let Some(cell) = self.cells.get(&coord.to_tuple_xy()) {
                        // calculate direction to store energy
                        (rec_coord, rec_dir) = Simulation::update_energy_dir(&self.world_map, &coord, cell, &self.cells);
                        if !self.cells.contains_key(&rec_coord.to_tuple_xy()) {
                            // kill cell (there is no receiver)
                            continue;
                        }
                        energy = cell.energy;
                    }
                    
                    // energy increase
                    let rec_cell = self.cells.get_mut(&rec_coord.to_tuple_xy());
                    if let Some(c) = rec_cell {
                        c.energy += energy;
                    } else {
                        panic!("There is not such cell! rec_coord is probably wrong here!");
                    }
                    // energy to zero
                    if let Some(c) = self.cells.get_mut(&coord.to_tuple_xy()) {
                        c.energy = 0f32;
                    }
                    // actual direction for energy flow set
                    let Some(cell) = self.cells.get_mut(&coord.to_tuple_xy()) else {
                        panic!("There is no cell with such coords!");
                    };
                    if let Some(d) = rec_dir { cell.out_dir = d; }
                    else { panic!("rec_dir is None for some reason here"); }
                },
                CellKind::Storage(s) => {
                    let Some(cell) = self.cells.get(&coord.to_tuple_xy()) else {
                        panic!("There is no cell with such coords {coord:?}!");
                    };
                    let slice_1 = Self::extract_window(&self.world_map.organics, &coord);
                    let slice_2 = Self::extract_window(&self.world_map.electric, &coord);

                    let input = Input {
                        organic_poisoning: slice_1.to_owned(),
                        electric_poisoning: slice_2.to_owned(),
                        energy: cell.energy
                    };
                    let actions = s.get_decision(input);
                    coord = Self::execute_actions(&mut self.cells, &mut new_coords, &self.world_map, actions, coord);
                    // println!("final bud coord is {coord:?}");
                    // println!("There is {} units of energy for now!", s.energy);
                },
            }
            // println!("coord is {coord:?}");

            // save this cell
            new_coords.push(coord.clone());
        }

        // for coord in new_coords.iter() {
        //     println!("coords in new coords: {coord:?}");
        //     assert!(self.cells.contains_key(&coord.to_tuple_xy()));
        // }

        self.coords = new_coords;
    }

    fn execute_actions(cells: &mut HashMap<(i64, i64), Cell>, 
                        new_coords: &mut Vec<Coord>, 
                        world_map: &Map, 
                        actions: Vec<Action>, 
                        coord: Coord) -> Coord {
        let mut final_bud_coord = coord.clone();
        let mut action_types: Vec<u8> = Vec::new();
        let mut need_energy = 0f32;
        let mut action_is_valid = [true; 4];
        let mut new_cells_coords: Vec<Coord> = Vec::new();

        let mut bud_counter = 0;
        let mut bud_dirs: Vec<Direction> = Vec::new();
        for (i, action) in actions.iter().enumerate() {
            action_types.push(action.1);
            let action_coord = coord.shift(&action.0);
            new_cells_coords.push(action_coord.clone());
            // вышли за рамки мира
            if !world_map.in_bounds(action_coord.x, action_coord.y) {
                action_is_valid[i] = false;
                continue;
            }
            // уже что-то есть - нельзя
            if cells.contains_key(&action_coord.to_tuple_xy()) {
                action_is_valid[i] = false;
                continue;
            }

            match action.1 {
                // codes: 
                // 0 - leaf
                // 1 - root
                // 2 - antena
                // 3 - move/create bud here
                0..3 => { need_energy += 0.1; },
                3 => { need_energy += 0.15; bud_counter += 1; bud_dirs.push(action.0.clone()); },
                _ => { panic!("there is not such code in my base!"); },
            }
        }
        let Some(cell) = cells.get(&coord.to_tuple_xy()) else {
            panic!("There is no cell with such coords!");
        };
        if need_energy > cell.energy { return final_bud_coord; }
        // there is some buds to create/move
        if bud_counter > 0 {
            // chose the main direction and new buds directions
            let mut r = rng();
            let main_bud_ind = r.random_range(0..bud_counter);
            // move main bud
            final_bud_coord = coord.shift(&bud_dirs[main_bud_ind]);
            new_coords.push(coord.clone());
            let conductor_cell = Cell {
                kind: CellKind::Conductor,
                life_time: 100,
                pos: coord.clone(),
                out_dir: bud_dirs[main_bud_ind].clone(),
                energy: 0f32
            };
            let Some(cell) = cells.remove(&coord.to_tuple_xy()) else {
                panic!("execute: there is not cell in this coords??");
            };
            cells.insert(coord.to_tuple_xy(), conductor_cell);
            cells.insert(final_bud_coord.to_tuple_xy(), cell);
            // create extra buds
            for i in 0..bud_counter {
                if i == main_bud_ind { continue; }
                let new_bud_coord = coord.shift(&bud_dirs[i]);
                new_coords.push(new_bud_coord.clone());
                let Some(cell) = cells.get(&final_bud_coord.to_tuple_xy()) else {
                    panic!("There is no cell with such coords!");
                };
                let genome = match &cell.kind {
                    CellKind::Storage(st) => {
                        if rng().random_bool(0.1) { st.genome.mutate() }
                        else { st.genome.clone() }
                    }
                    _ => { panic!("Is not bud cell here!!!"); }
                };
                let cell = Cell {
                    kind: CellKind::Storage(Storage {genome }),
                    life_time: 100,
                    pos: new_bud_coord.clone(),
                    out_dir: bud_dirs[i].clone(), // not important
                    energy: 0.15f32
                };
                cells.insert(new_bud_coord.to_tuple_xy(), cell);
            }
        }
        // create other cells
        // there is not buds to create/move
        for i in 0..actions.len() {
            // if 
            if !action_is_valid[i] { continue; }
            let kind = match actions[i].1 {
                0 => CellKind::Producer(Producer{resource: ResourceType::Solar}),
                1 => CellKind::Producer(Producer{resource: ResourceType::Organic}),
                2 => CellKind::Producer(Producer{resource: ResourceType::Electricity}),
                3 => { continue; },
                _ => panic!("there is not such code here"),
            };
            new_coords.push(new_cells_coords[i].clone());
            let out_dir = actions[i].0.oposite().clone();
            
            let cell = Cell {
                kind,
                life_time: 100,
                pos: new_cells_coords[i].clone(),
                out_dir,
                energy: 0f32
            };
            cells.insert(new_cells_coords[i].to_tuple_xy(), cell);
        }

        final_bud_coord
    }

    pub fn save_view(&self, overwrite: bool) -> std::io::Result<()> {
        // println!("saving...");

        let width = self.world_map.width as i64;
        let height = self.world_map.height as i64;

        let filename = format!("{}/{}_meta.txt", self.save_path, self.save_file_name);
        let path = Path::new(&filename);
        if !path.exists() || overwrite {
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)?;
            let mut w = BufWriter::new(f);
            // запись размеров: "width height"
            writeln!(w, "{},{}", self.world_map.height, self.world_map.width)?;
            w.flush()?;
        }

        let filename = format!("{}/{}_{}.csv", self.save_path, self.save_file_name, self.save_iter);
        let path = Path::new(&filename);

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut w = BufWriter::new(file);

        // записываем только занятые клетки: "x y"
        for (&(x, y), _cell) in &self.cells {
            writeln!(w, "{},{},{}", x, y, _cell.kind.str())?;
        }

        w.flush()?;
        Ok(())
    }
}