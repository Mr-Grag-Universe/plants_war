use std::collections::{HashMap};
use rand::{prelude::*, rng};
use ndarray::{s, Array2, SliceInfo, Dim, SliceInfoElem};
use rayon::iter::FromParallelIterator;
use std::cmp;
use std::fs::{OpenOptions, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::error::Error;

use crate::common::*;
use crate::map::{Map};
use crate::cells::*;


fn shuffled_indices(n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rand::rng();
    idx.shuffle(&mut rng);
    idx
}

struct SimulationSettings {
    life_time: i16,
    energy_expanse: HashMap<String, f32>,
    polution_increase: f32,
    polution_decrease: f32,

    polution_critical_lvl: f32,
}

impl SimulationSettings {
    fn save(&self, path: &Path, overwrite: bool) -> std::io::Result<()> {
        if !path.exists() || overwrite {
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)?;
            let mut w = BufWriter::new(f);
            // запись размеров: "width height"
            writeln!(w, "{}", self.life_time)?;
            writeln!(w, "{}", self.polution_increase)?;
            writeln!(w, "{}", self.polution_decrease)?;
            writeln!(w, "{}", self.polution_critical_lvl)?;
            for (key, val) in &self.energy_expanse {
                writeln!(w, "{}, {}", key, val)?;
            }
            w.flush()?;
        }

        Ok(())
    }

    fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        use std::io::BufRead;
        let f = File::open(path)?;
        let mut reader = std::io::BufReader::new(f);
        let mut line = String::new();

        // life_time
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err("simulation settings: missing life_time".into());
        }
        let life_time: i16 = line.trim().parse()?;

        // polution_increase
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err("simulation settings: missing polution_increase".into());
        }
        let polution_increase: f32 = line.trim().parse()?;

        // polution_decrease
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err("simulation settings: missing polution_decrease".into());
        }
        let polution_decrease: f32 = line.trim().parse()?;

        // polution_decrease
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err("simulation settings: missing polution_critical_lvl".into());
        }
        let polution_critical_lvl: f32 = line.trim().parse()?;

        // remaining lines: energy_expanse key, val pairs
        let mut energy_expanse: HashMap<String, f32> = HashMap::new();
        loop {
            line.clear();
            let bytes = reader.read_line(&mut line)?;
            if bytes == 0 {
                break;
            }
            let s = line.trim();
            if s.is_empty() {
                continue;
            }
            // expect "key, val" or "key,val"
            let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
            if parts.len() != 2 {
                return Err(format!("invalid energy_expanse line: {}", s).into());
            }
            let key = parts[0].to_string();
            let val: f32 = parts[1].parse()?;
            energy_expanse.insert(key, val);
        }

        Ok(SimulationSettings {
            life_time,
            energy_expanse,
            polution_increase,
            polution_decrease,
            polution_critical_lvl,
        })
    }
}


pub struct Simulation {
    cells: HashMap<(i64,i64), Cell>,
    world_map: Map,

    
    pub save_iter: usize,
    save_path: String,
    save_file_name: String,

    settings: SimulationSettings,
}

impl Simulation {
    fn get_energy_expanse() -> HashMap<String, f32> {
        let x = HashMap::<String, f32>::from_par_iter([
            (String::from("producer"), 0.1),
            (String::from("storage"), 0.25),
        ]);
        return x;
    }

    pub fn new(world_map: Option<Map>, 
               save_path: String, save_file_name: String,
               life_time: i16) -> Self {
        let world_map: Map = world_map.unwrap_or_else(|| Map::new(1024, 1024));
        let cells: HashMap<(i64,i64), Cell> = HashMap::new();
        let save_iter = 0;
        
        let energy_expanse = Self::get_energy_expanse();
        let settings = SimulationSettings {
            life_time,
            energy_expanse,
            polution_increase: 0.1f32, // useless
            polution_decrease: 0.1f32, // useless
            polution_critical_lvl: 15f32
        };

        Simulation { 
            cells, 
            world_map, 
            save_iter,
            save_path,
            save_file_name,
            settings,
        }
    }

    const WIN_W: usize = 5;
    const WIN_H: usize = 5;
    const PAD_VALUE: f32 = -1.0; // или 0.0
    const HW: usize = Self::WIN_W / 2;
    const HH: usize = Self::WIN_H / 2;

    fn extract_window(
        map: &Array2<f32>, // или другой тип элемента
        coord: &Coord,      // { x: usize, y: usize }
    ) -> Array2<f32> {
        // центральное окно размером WIN_H x WIN_W, с центром в coord
        // в вашем примере вы брали от x-3..x+4 (7 ширина) и y-3..y+4
        let map_h = map.nrows();
        let map_w = map.ncols();

        let src_x0 = coord.x.saturating_sub(Self::HW as i64) as usize;
        let src_y0 = coord.y.saturating_sub(Self::HH as i64) as usize;
        let src_x1 = (coord.x as usize + Self::HW + 1).min(map_w); // exclusive
        let src_y1 = (coord.y as usize + Self::HH + 1).min(map_h);

        let mut out = Array2::from_elem((Self::WIN_H, Self::WIN_W), Self::PAD_VALUE);

        if src_x0 < src_x1 && src_y0 < src_y1 {
            let dst_x0 = Self::HW.saturating_sub(coord.x.saturating_sub(src_x0 as i64) as usize);
            let dst_y0 = Self::HH.saturating_sub(coord.y.saturating_sub(src_y0 as i64) as usize);
            let dst_x1 = dst_x0 + (src_x1 - src_x0);
            let dst_y1 = dst_y0 + (src_y1 - src_y0);

            let src_view = map.slice(s![src_y0..src_y1, src_x0..src_x1]);
            out.slice_mut(s![dst_y0..dst_y1, dst_x0..dst_x1]).assign(&src_view);
        }

        out
    }

    pub fn add_cells(&mut self, cells: Vec<Cell>) {
        for cell in cells {
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
                        CellKind::Producer(_) => continue,
                        _ => {
                            // new receiver found
                            rec_coord = rec_coord_tmp;
                            rec_dir = Some(dir.clone());
                            break;
                        }
                    }
                }
            }
            // if let Some(d) = &rec_dir { println!("contains!"); }
            // else { println!("Did not find!"); }
        }
        (rec_coord, rec_dir)
    }

    fn get_slice(center_coord : &Coord, r: i64, h: usize, w: usize) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
        let l_x = cmp::max(center_coord.x-r, 0) as usize;
        let t_y = cmp::max(center_coord.y-r, 0) as usize;
        let r_x = cmp::min(center_coord.x+r+1, w as i64) as usize;
        let b_y = cmp::min(center_coord.y+r+1, h as i64) as usize;
        s![t_y..b_y, l_x..r_x]
    }

    pub fn get_coords(&self) -> Vec<Coord> {
        let mut coords: Vec<Coord> = Vec::with_capacity(self.cells.len());
        for ((x, y), _) in &self.cells {
            coords.push(Coord { x: *x, y: *y });
        }
        coords
    }

    pub fn step(&mut self) {
        let coords: Vec<Coord> = self.get_coords();
        let order = shuffled_indices(self.cells.len());

        // decrease life time of all existing cells
        for (_, cell) in self.cells.iter_mut() {
            cell.life_time -= 1;
        }

        let mut deleted_cells_count = 0;
        for i in order {
            let coord = coords[i].clone();
            let key = coord.to_tuple_xy();
            if !self.cells.contains_key(&key) { continue; }

            let kind = {
                let Some(cell) = self.cells.get(&key) else {
                    panic!("There is no cell with such coords {coord:?}!");
                };
                
                // if it's dead
                if cell.life_time <= 0 {
                    self.cells.remove(&key);
                    Self::increase_polution(&mut self.world_map, &coord);
                    deleted_cells_count += 1;
                    // println!("delete from coord: (x={}, y={})", coord.x, coord.y);
                    continue;
                }

                &cell.kind
            };

            // check if it's too poluted to live here
            match kind {
                CellKind::Conductor => {
                    let (org, elc) = self.world_map.is_lvl_critical(
                        coord.x as usize, coord.y as usize, 
                        self.settings.polution_critical_lvl);
                    if org || elc {
                        self.cells.remove(&key);
                        continue;
                    }
                },
                CellKind::Producer(p) => {
                    let (org, elc) = self.world_map.is_lvl_critical(
                        coord.x as usize, coord.y as usize, 
                        self.settings.polution_critical_lvl);
                    match p.resource {
                        ResourceType::Solar => {
                            if org || elc { self.cells.remove(&key); continue; }
                        },
                        ResourceType::Organic => {
                            if elc { self.cells.remove(&key); continue; }
                        },
                        ResourceType::Electricity => {
                            if org { self.cells.remove(&key); continue; }
                        },
                    }
                },
                _ => {}
            }

            match kind {
                CellKind::Producer(p) => {
                    // calculate direction to store energy
                    let (rec_coord, rec_dir) = Simulation::update_energy_dir(&self.world_map, &coord, &self.cells[&key], &self.cells);
                    if !self.cells.contains_key(&rec_coord.to_tuple_xy()) {
                        // kill cell (there is no receiver)
                        self.cells.remove(&key);
                        continue;
                    }
                    // println!("Producer cell: {:?} {:?}", rec_coord, rec_dir);

                    // produced energy
                    let mut energy_produced = 0f32;
                    match p.resource {
                        ResourceType::Solar => energy_produced = 0.1f32,
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
                    
                    
                    let rec_key = rec_coord.to_tuple_xy();
                    let rec_cell = match self.cells.get_mut(&rec_key) {
                        Some(c) => c,
                        None => panic!(), // no receiver — skip
                    };
                    rec_cell.energy += energy_produced;

                    // actual direction for energy flow set
                    if let Some(cell_mut) = self.cells.get_mut(&key) {
                        cell_mut.out_dir = rec_dir.expect("rec_dir is None");
                    } else {
                        panic!("There is no cell with such coords  {coord:?}!");
                    }
                },
                CellKind::Conductor => {
                    let (rec_coord, rec_dir) = {
                        let cell_ref = &self.cells[&key];
                        Simulation::update_energy_dir(&self.world_map, &coord, cell_ref, &self.cells)
                    };
                    if !self.cells.contains_key(&rec_coord.to_tuple_xy()) {
                        // kill cell (there is no receiver)
                        self.cells.remove(&key);
                        continue;
                    }
                    let rec_key = rec_coord.to_tuple_xy();
                    let energy = self.cells[&key].energy;
                    
                    // energy increase
                    if let Some(c) = self.cells.get_mut(&rec_key) {
                        c.energy += energy;
                    } else { panic!(); }
                    // energy to zero
                    if let Some(c) = self.cells.get_mut(&coord.to_tuple_xy()) {
                        c.energy = 0f32;
                    } else { panic!(); }
                    // actual direction for energy flow set
                    if let Some(c) = self.cells.get_mut(&key) {
                        c.energy = 0.0;
                        c.out_dir = rec_dir.expect("rec_dir is None");
                    } else { panic!(); }
                },
                CellKind::Storage(s) => {
                    let Some(cell_ref) = self.cells.get(&key) else { panic!() };

                    let slice_1 = Self::extract_window(&self.world_map.organics, &coord);
                    let slice_2 = Self::extract_window(&self.world_map.electric, &coord);

                    let input = Input {
                        organic_poisoning: slice_1.to_owned(),
                        electric_poisoning: slice_2.to_owned(),
                        energy: cell_ref.energy
                    };
                    let actions = s.get_decision(input);
                    Self::execute_actions(&mut self.cells, &self.world_map, actions, coord, &self.settings);
                },
            }
        }
        // println!("Cells count: {}, Coodrs count: {}", self.cells.len(), new_coords.len());
        // println!("Deleted cells count: {}", deleted_cells_count);
    }

    fn execute_actions(cells: &mut HashMap<(i64, i64), Cell>, 
                        world_map: &Map, 
                        actions: Vec<Action>, 
                        coord: Coord, settings: &SimulationSettings) -> Coord {
        let mut final_bud_coord = coord.clone();
        let mut need_energy = 0f32;
        let mut action_is_valid = [true; 4];
        let mut new_cells_coords: Vec<Coord> = Vec::new();

        let mut bud_counter = 0;
        let mut bud_dirs: Vec<Direction> = Vec::new();

        for (i, action) in actions.iter().enumerate() {
            let action_coord = coord.shift(&action.0);
            let action_coord_key = action_coord.to_tuple_xy();
            new_cells_coords.push(action_coord.clone());

            // вышли за рамки мира
            if !world_map.in_bounds(action_coord.x, action_coord.y) {
                action_is_valid[i] = false;
                continue;
            }
            
            if cells.contains_key(&action_coord_key) {
                match action.1 {
                    3 => {
                        // bud может съесть
                        let cell_killed_energy = cells.get(&action_coord_key).expect("cannot be None").energy;
                        let cell_hunter = cells.get_mut(&coord.to_tuple_xy()).expect("cannot be None");
                        cell_hunter.energy += cell_killed_energy * 0.7;
                        // delete cell from world
                        cells.remove(&action_coord_key).expect("there was not cell there");
                    },
                    _ => {
                        // уже что-то есть - нельзя
                        action_is_valid[i] = false;
                        continue;
                    }
                }
            }

            match action.1 {
                // codes: 
                // 0 - leaf
                // 1 - root
                // 2 - antena
                // 3 - move/create bud here
                0..=2 => { need_energy += settings.energy_expanse["producer"]; },
                3 => { 
                    need_energy += settings.energy_expanse["storage"]; 
                    bud_counter += 1; 
                    bud_dirs.push(action.0.clone()); 
                },
                _ => unreachable!("unknown action code"),
            }
        }

        let cell_key = coord.to_tuple_xy();
        let cell = match cells.get(&cell_key) {
            Some(c) => c,
            None => panic!("There is no cell with such coords!"),
        };
        if need_energy > cell.energy { return final_bud_coord; }
        
        // there is some buds to create/move
        if bud_counter > 0 {
            // chose the main direction and new buds directions
            let mut r = rng();
            let main_bud_ind = r.random_range(0..bud_counter);

            // move main bud
            final_bud_coord = coord.shift(&bud_dirs[main_bud_ind]);
            
            let conductor = Cell {
                kind: CellKind::Conductor,
                life_time: settings.life_time,
                pos: coord.clone(),
                out_dir: bud_dirs[main_bud_ind].clone(),
                energy: 0f32
            };

            let old_cell = cells.remove(&cell_key).expect("execute: there is not cell in this coords??");
            cells.insert(cell_key, conductor);
            cells.insert(final_bud_coord.to_tuple_xy(), old_cell);

            // create extra buds
            for i in 0..bud_counter {
                if i == main_bud_ind { continue; }
                let new_bud_coord = coord.shift(&bud_dirs[i]);

                let parent_key = final_bud_coord.to_tuple_xy();
                let parent_cell = cells.get(&parent_key).expect("There is no cell with such coords.");
                let genome = match &parent_cell.kind {
                    CellKind::Storage(st) => {
                        if rng().random_bool(0.15) { st.genome.mutate() }
                        else { st.genome.clone() }
                    }
                    _ => { panic!("Is not bud cell here!!!"); }
                };
                
                let new_cell = Cell {
                    kind: CellKind::Storage(Storage { genome }),
                    life_time: settings.life_time,
                    pos: new_bud_coord.clone(),
                    out_dir: bud_dirs[i].clone(),
                    energy: settings.energy_expanse["storage"]*0.8
                };
                cells.insert(new_bud_coord.to_tuple_xy(), new_cell);
            }
        }

        // create other cells
        // there is not buds to create/move
        for (i, action) in actions.iter().enumerate() {
            if !action_is_valid[i] { continue; }
            let kind = match action.1 {
                0 => CellKind::Producer(Producer{resource: ResourceType::Solar}),
                1 => CellKind::Producer(Producer{resource: ResourceType::Organic}),
                2 => CellKind::Producer(Producer{resource: ResourceType::Electricity}),
                3 => continue,
                _ => unreachable!("unknown action code"),
            };
            let pos = new_cells_coords[i].clone();
            let out_dir = action.0.oposite().clone();
            
            let cell = Cell {
                kind,
                life_time: settings.life_time,
                pos,
                out_dir,
                energy: 0f32
            };
            cells.insert(cell.pos.to_tuple_xy(), cell);
        }

        final_bud_coord
    }

    pub fn save_view(&self, overwrite: bool) -> std::io::Result<()> {
        // println!("saving...");
        ensure_dir(Path::new(&self.save_path)).expect("Cannot ensure save directory!");
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

    pub fn save_state(&self, overwrite: bool) -> std::io::Result<()> {
        let save_path = format!("{}_back", self.save_path);
        ensure_dir(Path::new(&save_path)).expect("save_state: Cannot ensure save directory!");
        
        // save map info
        let map_path = Path::new(save_path.as_str()).join("map");
        ensure_dir(Path::new(&map_path)).expect("save_state: Cannot ensure save directory!");
        self.world_map.save(map_path.as_path(), overwrite).expect("cannot save map to file!");

        // simulation dir
        let sim_path = Path::new(save_path.as_str()).join("sim");
        ensure_dir(Path::new(&sim_path)).expect("save_state: Cannot ensure save directory!");

        // save meta info about simulation
        let path = sim_path.join("simulation_meta.txt");
        if !path.exists() || overwrite {
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)?;
            let mut w = BufWriter::new(f);
            // запись размеров: "width height"
            writeln!(w, "{},{}", self.world_map.height, self.world_map.width)?;
            writeln!(w, "{}", self.save_iter)?;
            writeln!(w, "{}", self.save_path)?;
            writeln!(w, "{}", self.save_file_name)?;
            w.flush()?;
        }

        // save simulation settings
        let path = sim_path.join("simulation_settings.txt");
        self.settings.save(path.as_path(), overwrite).expect("cannot save simulation settings!!!");

        // save cells
        let path = sim_path.join("cells");
        ensure_dir(path.as_path()).expect("save_state: Cannot ensure save directory!");
        self.save_cells(path.as_path(), overwrite).expect("cannot save cells!!!");

        Ok(())
    }

    fn save_cells(&self, path: &Path, overwrite: bool) -> std::io::Result<()> {
        for (i, (coord, cell)) in self.cells.iter().enumerate() {
            let cell_path = path.join(format!("cell_{}", i));
            ensure_dir(cell_path.as_path()).expect("save_state: Cannot ensure save directory!");
            cell.save(cell_path.as_path(), overwrite).expect("save_state: cannot save cell to cell_dir");

            // save_coords
            let coord_path: std::path::PathBuf = cell_path.join("coord.txt");
            if !coord_path.exists() || overwrite {
                let f = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(coord_path)?;
                let mut w = BufWriter::new(f);
                // запись размеров: "width height"
                writeln!(w, "{},{}", coord.0, coord.1)?;
                w.flush()?;
            }
        }
        Ok(())
    }

    pub fn load(save_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        // load map
        let map_path = save_path.join("map");
        let world_map = Map::load(&map_path)?;

        // load sim meta
        let sim_path = save_path.join("sim");
        let meta_path = sim_path.join("simulation_meta.txt");
        let mut save_iter = 0usize;
        let mut save_path_str = String::new();
        let mut save_file_name = String::new();
        if meta_path.exists() {
            let f = File::open(&meta_path)?;
            let mut br = BufReader::new(f);
            let mut lines = Vec::new();
            loop {
                let mut tmp = String::new();
                let bytes = br.read_line(&mut tmp)?;
                if bytes == 0 { break; }
                lines.push(tmp);
            }
            if !lines.is_empty() {
                let dims: Vec<&str> = lines[0].trim().split(',').collect();
                if dims.len() == 2 {
                    // optionally use dims
                }
            }
            if lines.len() > 1 {
                save_iter = lines[1].trim().parse().unwrap_or(0);
            }
            if lines.len() > 2 {
                save_path_str = lines[2].trim().to_string();
            }
            if lines.len() > 3 {
                save_file_name = lines[3].trim().to_string();
            }
        }

        // load settings
        let settings_path = sim_path.join("simulation_settings.txt");
        let settings = SimulationSettings::load(&settings_path)?;

        // load cells
        let cells_path = sim_path.join("cells");
        let cells = Self::load_cells(&cells_path)?;

        Ok(Self {
            world_map,
            settings,
            cells,
            save_iter,
            save_path: save_path_str,
            save_file_name
        })
    }

    fn load_cells(path: &std::path::Path) -> Result<HashMap<(i64, i64), Cell>, Box<dyn std::error::Error>> {
        if !path.exists() {
            panic!("there is not saved cells here!");
        }
        let mut cells: HashMap<(i64, i64), Cell> = HashMap::new();
        // перебираем папки cell_*
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() { continue; }
            let cell_dir = entry.path();
            // load coord
            let coord_path = cell_dir.join("coord.txt");
            let coord = if coord_path.exists() {
                let f = File::open(coord_path)?;
                let mut br = BufReader::new(f);
                let mut line = String::new();
                br.read_line(&mut line)?;
                let parts: Vec<&str> = line.trim().split(',').collect();
                if parts.len() != 2 { return Err("coord.txt malformed".into()); }
                let x: i64 = parts[0].parse()?;
                let y: i64 = parts[1].parse()?;
                Coord { x, y }
            } else {
                // fallback: try parse index from folder name
                let name = entry.file_name().into_string().unwrap_or_default();
                // координаты 0,0 по умолчанию
                Coord { x: 0, y: 0 }
            };
            let cell = Cell::load(&cell_dir)?;
            cells.insert(coord.to_tuple_xy(), cell);
        }
        Ok(cells)
    }
}