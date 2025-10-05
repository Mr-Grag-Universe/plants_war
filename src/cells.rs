use crate::common::*;
use ndarray::{ArrayView1, Array1, Array2, Axis, s, Ix1};
use ndarray;
use rand::rng;
use rand_distr::{Distribution, Normal};
use std::path::Path;
use std::error::Error;
use std::fs::{OpenOptions, File};
use std::io::{BufWriter, Write, BufReader, BufRead};

// #[derive(Debug)]
pub struct Genome {
    w1: Array2<f32>,
    w2: Array2<f32>,
    w3: Array2<f32>,
    pub activation: Box<dyn Fn(&Array1<f32>) -> Array1<f32> + Sync + Send>,
}
impl Genome {
    /// Создать Genome с матрицами указанной формы и инициализировать
    /// веса нормальным распределением N(mean, std).
    pub fn random(
        n_in: usize,
        n_hidden1: usize,
        n_hidden2: usize,
        n_out: usize,
        mean: f32,
        std: f32,
    ) -> Self {
        let mut rng = rng();
        let normal = Normal::new(mean, std).unwrap();

        let total_w1 = n_hidden1 * n_in;
        let total_w2 = n_hidden2 * n_hidden1;
        let total_w3 = n_out * n_hidden2;

        let w1_vec: Vec<f32> = (0..total_w1).map(|_| normal.sample(&mut rng)).collect();
        let w2_vec: Vec<f32> = (0..total_w2).map(|_| normal.sample(&mut rng)).collect();
        let w3_vec: Vec<f32> = (0..total_w3).map(|_| normal.sample(&mut rng)).collect();

        let w1 = Array2::from_shape_vec((n_hidden1, n_in), w1_vec).unwrap();
        let w2 = Array2::from_shape_vec((n_hidden2, n_hidden1), w2_vec).unwrap();
        let w3 = Array2::from_shape_vec((n_out, n_hidden2), w3_vec).unwrap();

        // ReLU in-place would be more efficient, но для совместимости возвращаем новый Array1
        let relu = Box::new(|v: &Array1<f32>| -> Array1<f32> {
            v.mapv(|x| if x > 0.0 { x } else { 0.0 })
        });

        Genome { w1, w2, w3, activation: relu }
    }

    pub fn clone(&self) -> Self {
        let activation = Box::new(|v: &Array1<f32>| -> Array1<f32> {
            v.mapv(|x| if x > 0.0 { x } else { 0.0 })
        });

        Genome {
            w1: self.w1.clone(),
            w2: self.w2.clone(),
            w3: self.w3.clone(),
            activation,
        }
    }

    pub fn mutate(&self) -> Self {
        let mut rng = rng();
        let mutation_std = 0.1;
        let normal = Normal::new(0.0, mutation_std).unwrap();

        // W1
        let len1 = self.w1.len();
        let mut noise1: Vec<f32> = Vec::with_capacity(len1);
        for _ in 0..len1 {
            noise1.push(normal.sample(&mut rng));
        }
        let noise1_arr = Array2::from_shape_vec(self.w1.raw_dim(), noise1).unwrap();
        let new_w1 = &self.w1 + &noise1_arr;

        // W2
        let len2 = self.w2.len();
        let mut noise2: Vec<f32> = Vec::with_capacity(len2);
        for _ in 0..len2 {
            noise2.push(normal.sample(&mut rng));
        }
        let noise2_arr = Array2::from_shape_vec(self.w2.raw_dim(), noise2).unwrap();
        let new_w2 = &self.w2 + &noise2_arr;

        // W3
        let len3 = self.w3.len();
        let mut noise3: Vec<f32> = Vec::with_capacity(len3);
        for _ in 0..len3 {
            noise3.push(normal.sample(&mut rng));
        }
        let noise3_arr = Array2::from_shape_vec(self.w3.raw_dim(), noise3).unwrap();
        let new_w3 = &self.w3 + &noise3_arr;

        let activation = Box::new(|v: &Array1<f32>| -> Array1<f32> {
            v.mapv(|x| if x > 0.0 { x } else { 0.0 })
        });

        Genome {
            w1: new_w1,
            w2: new_w2,
            w3: new_w3,
            activation,
        }
    }
}


pub struct Input {
    pub organic_poisoning: Array2<f32>,
    pub electric_poisoning: Array2<f32>,
    pub energy: f32,
}
impl Input {
    pub fn flatten(&self) -> Array1<f32> {
        let a_view: ArrayView1<f32> =
            self.organic_poisoning.view().into_shape_with_order(self.organic_poisoning.len()).unwrap();
        let b_view: ArrayView1<f32> =
            self.electric_poisoning.view().into_shape_with_order(self.electric_poisoning.len()).unwrap();
        let e = Array1::from_vec(vec![self.energy]);
        // println!("input: ");
        // println!("{}, {}, {}", a_view, b_view, e);
        ndarray::concatenate(Axis(0), &[a_view, b_view, e.view()]).unwrap()
    }
}

#[derive(Debug)]
pub struct Producer {
    pub resource: ResourceType,
}

// #[derive(Debug)]
// pub struct Conductor {
//     out_dir: Direction,
//     energy: f32,
// }

// #[derive(Debug)]
pub struct Storage {
    pub genome: Genome,
}
impl Storage {
    pub fn get_decision(&self, input: Input) -> Vec<Action> {
        let input_arr = input.flatten();

        // calculating NN result
        let mut out: Array1<f32> = self.genome.w1.dot(&input_arr);
        out = (self.genome.activation)(&out);
        out = self.genome.w2.dot(&out);
        out = (self.genome.activation)(&out);
        let mat = self.genome.w3.dot(&out);
        let flat: Array1<f32> = match mat.clone().into_dimensionality::<Ix1>() {
            Ok(v) => v,
            Err(_) => {
                let v: Vec<f32> = mat.iter().cloned().collect();
                if v.len() != 5*4 {
                    panic!("expected length 20, got {}", v.len());
                }
                Array1::from(v)
            }
        };
        out = flat;

        // codes: 
        // 0 - leaf
        // 1 - root
        // 2 - antena
        // 3 - move bud here

        let mut actions: Vec<Action> = Vec::new();
        for i in 0..4 {
            let action_p = out.slice(s![(i*4)..((i+1)*4)]);
            if action_p[0] > 0f32 {
                let action_type_p = out.slice(s![4..4*2]);
                let argmax = action_type_p
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .map(|i| i as u8);
                if let Some(argmax) = argmax {
                    actions.push(Action(Direction::all_directions()[i].clone(), argmax));
                }
            }
        }
        
        actions
    }
}

// #[derive(Debug)]
pub enum CellKind {
    Producer(Producer),
    Conductor,
    Storage(Storage),
}

impl CellKind {
    pub fn str(&self) -> &str {
        match self {
            Self::Producer(_) => "producer",
            Self::Storage(_)  => "bud",
            Self::Conductor   => "conductor",
        }
    }
}

// #[derive(Debug)]
pub struct Cell {
    pub kind: CellKind,
    pub life_time: i16,
    pub pos: Coord,
    pub out_dir: Direction,
    pub energy: f32,
}

impl Cell {
    pub fn save(&self, save_path: &Path, overwrite: bool) -> Result<(), Box<dyn Error>> {
        // main.txt (meta)
        let meta_path = save_path.join("main.txt");
        if !meta_path.exists() || overwrite {
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&meta_path)?;
            let mut w = BufWriter::new(f);
            // Пример: kind, life_time, pos(x,y), out_dir, energy
            writeln!(
                w,
                "{},{},{},{}",
                self.kind.str(),
                self.life_time,
                self.pos.x, // предполагается, что Coord имеет поля x,y
                self.pos.y
            )?;
            writeln!(w, "out_dir:{:?}", self.out_dir)?;
            writeln!(w, "energy:{}", self.energy)?;
            w.flush()?;
        }

        // genomes directory
        let genomes_dir = save_path.join("genomes");
        ensure_dir(genomes_dir.as_path()).expect("cannot ensure genome directory");

        // If cell has a Genome (Storage), save matrices
        if let CellKind::Storage(storage) = &self.kind {
            let genome_dir = genomes_dir;
            ensure_dir(genome_dir.as_path()).expect("cannot ensure storage directory");

            let w1_path = genome_dir.join("w1.npy");
            save_npy(&storage.genome.w1, w1_path.as_path())?;
            let w2_path = genome_dir.join("w2.npy");
            save_npy(&storage.genome.w2, w2_path.as_path())?;
            let w3_path = genome_dir.join("w3.npy");
            save_npy(&storage.genome.w3, w3_path.as_path())?;
            // optionally save activation choice as text
        }

        // If Producer or other kinds, save their data if needed
        if let CellKind::Producer(prod) = &self.kind {
            let prod_path = save_path.join("producer.txt");
            let f = File::create(prod_path)?;
            let mut bw = BufWriter::new(f);
            writeln!(bw, "resource:{:?}", prod.resource)?;
        }

        Ok(())
    }

    pub fn load(save_path: &Path) -> Result<Self, Box<dyn Error>> {
        // main.txt
        let meta_path = save_path.join("main.txt");
        if !meta_path.exists() {
            return Err(format!("main.txt not found in {:?}", save_path).into());
        }
        let f = File::open(&meta_path)?;
        let mut reader = BufReader::new(f);
        let mut line = String::new();

        // First line: "{kind},{life_time},{pos.x},{pos.y}"
        reader.read_line(&mut line)?;
        if line.trim().is_empty() {
            return Err("main.txt is empty".into());
        }
        let first = line.trim().to_string();
        let parts: Vec<&str> = first.split(',').collect();
        if parts.len() < 4 {
            return Err(format!("unexpected main.txt first line format: {}", first).into());
        }
        let kind_str = parts[0];
        let life_time: i16 = parts[1].parse()?;
        let pos_x: i64 = parts[2].parse()?; // предполагаем целочисленные координаты
        let pos_y: i64 = parts[3].parse()?;
        let pos = Coord { x: pos_x, y: pos_y };

        // Second line: "out_dir:{:?}"
        line.clear();
        reader.read_line(&mut line)?;
        let out_dir = if !line.trim().is_empty() {
            let s = line.trim();
            // ожидаем формат "out_dir:Direction::Something" или "out_dir:Something" или "out_dir:Direction({:?})"
            // Попробуем извлечь часть после ':' и убрать префиксы/скобки
            if let Some(idx) = s.find(':') {
                let v = s[(idx + 1)..].trim();
                // Попробуем парсить через FromStr для Direction, иначе через Debug string
                if let Ok(d) = Direction::from_str(v) {
                    d
                } else {
                    // Удалим возможные символы "Direction", '"' и т.д.
                    let cleaned = v.replace("Direction::", "")
                                   .replace("Direction(", "")
                                   .replace(')', "")
                                   .replace('"', "")
                                   .trim()
                                   .to_string();
                    Direction::from_str(&cleaned)?
                }
            } else {
                return Err("cannot parse out_dir line".into());
            }
        } else {
            return Err("out_dir line missing".into());
        };

        // Third line: "energy:{}"
        line.clear();
        reader.read_line(&mut line)?;
        let energy: f32 = if !line.trim().is_empty() {
            let s = line.trim();
            if let Some(idx) = s.find(':') {
                let v = s[(idx + 1)..].trim();
                v.parse()?
            } else {
                return Err("cannot parse energy line".into());
            }
        } else {
            0.0
        };

        // Now determine kind and load additional data
        let genomes_dir = save_path.join("genomes");
        let kind = match kind_str {
            "producer" => {
                // try read producer.txt for resource
                let prod_path = save_path.join("producer.txt");
                let resource = if prod_path.exists() {
                    let pf = File::open(prod_path)?;
                    let mut pr = BufReader::new(pf);
                    let mut prod_line = String::new();
                    pr.read_line(&mut prod_line)?;
                    // ожидаем "resource:{:?}"
                    if let Some(idx) = prod_line.find(':') {
                        let val = prod_line[(idx + 1)..].trim();
                        // Попробуем парсить через FromStr для ResourceType
                        if let Ok(r) = ResourceType::from_str(val) {
                            r
                        } else {
                            // убрать "ResourceType::" и т.д.
                            let cleaned = val.replace("ResourceType::", "").replace('"', "").trim().to_string();
                            ResourceType::from_str(&cleaned)?
                        }
                    } else {
                        return Err("cannot parse producer.txt".into());
                    }
                } else {
                    // default resource если нет файла
                    ResourceType::default()
                };
                CellKind::Producer(Producer { resource })
            }
            "bud" | "storage" => {
                // load genome from .npy files
                if !genomes_dir.exists() {
                    return Err(format!("genomes directory not found in {:?}", save_path).into());
                }
                let w1_path = genomes_dir.join("w1.npy");
                let w2_path = genomes_dir.join("w2.npy");
                let w3_path = genomes_dir.join("w3.npy");
                if !w1_path.exists() || !w2_path.exists() || !w3_path.exists() {
                    return Err("one of genome files (w1.npy,w2.npy,w3.npy) is missing".into());
                }

                // предполагается функция load_npy -> Array2<f32>
                let w1: Array2<f32> = load_npy(&w1_path)?;
                let w2: Array2<f32> = load_npy(&w2_path)?;
                let w3: Array2<f32> = load_npy(&w3_path)?;

                let activation = Box::new(|v: &Array1<f32>| -> Array1<f32> {
                    v.mapv(|x| if x > 0.0 { x } else { 0.0 })
                });

                let genome = Genome { w1, w2, w3, activation };
                CellKind::Storage(Storage { genome })
            }
            "conductor" => CellKind::Conductor,
            other => {
                return Err(format!("unknown cell kind: {}", other).into());
            }
        };

        Ok(Cell {
            kind,
            life_time,
            pos,
            out_dir,
            energy,
        })
    }
}