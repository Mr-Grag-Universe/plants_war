use crate::common::*;
use ndarray::{ArrayView1, Array1, Array2, Axis, s, Ix1};
use ndarray;
use rand::rng;
use rand_distr::{Distribution, Normal};

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