use rand::Rng;
use neural_net::utils::{self, sigmoid, sigmoid_derivative};
use ndarray::{Array1, Array2, ArrayView1};
use std::{sync::Arc, time::Instant};


#[derive(Debug)]
struct NeuralNet {
    layers: Vec<Layer>
}

impl NeuralNet {
    fn new(layer_sizes: &[usize]) -> NeuralNet {
        // iterating over the layer array input by window of 2 we create new layers
        let layers: Vec<Layer> = layer_sizes.windows(2)
            .map(|w: &[usize]| Layer::new(w[0],w[1]))
            .collect();
        NeuralNet { layers }
    }

    fn forward(&self, input: &[f64]) -> Vec<Vec<f64>> {
        // as each forward pass returns a vector of values, that vector in turn is used to propagate forward
        // this propagation forward vector is then used to multiple against the weights called forth
        // it is important to remember that the layers are defined as the vectors of weights and biases NOT the node values
        let mut activations = Vec::new();
        let current = input.to_vec();
        activations.push(current);

        self.layers.iter().fold(input.to_vec(), |acc, layer| {
            let activation = layer.forward(&acc);

            activations.push(activation.clone().to_vec());
            activation.to_vec()
            
        });

        activations
    }

    fn cycle(&self, input: &[f64], target: &[f64], learning_rate: f64) -> () {
        
        // get activations from forward pass
        let activations = self.nd_forward(input);
        
        println!("{:?} \n\n",activations);

        // retrive last layer activation values to determine error
        let output = activations.last().expect("Last vector was empty");

        // compute the deltas from final activation
        let error: Array1<f64> = output.iter().zip(target.to_vec()).map(|(a, y)| a - y).collect();

        // pre-allocate array of deltas
        let mut deltas: Vec<Array1<f64>> = self.layers.iter()
            // confirm this is accurate... shape()[1] it mmight have to take on the full shape...
            .map(|layer| Array1::zeros(layer.weights.shape()[1]))
            .collect();

        
        
        // calculate delta for last layer
        deltas.last_mut().expect("Last vector was empty")
            .assign(
                &(error * activations.last().expect("last vector was empty").mapv(sigmoid_derivative))
            );
        


        self.layers.iter().rev().skip(1)
            .zip(self.layers.iter().rev())
            .enumerate()
            .for_each(|(l, (_current_layer,next_layer))| {
                // TODO leverage split_at_mut on deltas to obtain two separate slices
                // what is done below is not a memory useful methodology
                let delta2 = deltas.clone();
                println!("{:?}",deltas);
                println!("\n\n");
                
                deltas[l].assign( {

                    &(next_layer.weights.dot(&delta2[l+1]) * &activations[l+1].mapv(sigmoid_derivative))
                })
            }
        );

        for x in deltas {
            println!("{:?}",x);
        }
    }

    fn nd_forward(&self, input: &[f64]) -> Vec<Array1<f64>> {
        let mut activations:Vec<Array1<f64>> = Vec::new();
        activations.push(Array1::from_vec(input.to_vec()));

        self.layers.iter().fold(input.to_vec(), |acc: Vec<f64>, layer| {
            let accv: Vec<f64> = acc.to_vec();
            let activation = layer.forward(&accv);

            activations.push(activation.clone());
            activation.to_vec()
        });


        activations
    }
}

#[derive(Debug)]
struct Layer {
    weights: Array2<f64>,
    biases: Array1<f64>
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Layer {
        let mut rng = rand::thread_rng();

        //Initialize the weights and biases randomly
        let weights: Array2<f64> = Array2::from_shape_simple_fn((input_size,output_size), || {rng.gen_range(-1.0..1.0)});

        let biases: Array1<f64> = (0..output_size)
            .map(|_| rng.gen_range(-1.0..1.0)).collect();

        Layer { weights, biases}
    }
    
    // fn forward_manual(&self, input: &[f64]) -> Vec<f64> {
        
    //     self.weights.iter().enumerate().map(|(i, neuron_weights)| {

    //         // iterating over each neuron, multiply the weight & the input value and add the bias
    //         // then apply the sigmoid function to flatten TODO: can make this a parameterized call based on sigmoid or ReLu later
    //         let sum: f64 = neuron_weights.iter().zip(input.iter())
    //             .map(|(w,i)| w * i)
    //             .sum();
    //         sigmoid(sum + self.biases[i])
    //     }).collect()    
    // }

    fn forward(&self, input: &[f64]) -> Array1<f64> {

        // Interestingly... this can be done without copying using ArrayView1 feature from ndarray, which could be faster
        let input_array = ArrayView1::from(input);

        // Or done with copying, which increases memory OH
        // let input_array = Array1::from(input.to_vec());

        // Perform matrix-vector multiplication and add biases
        let z = self.weights.t().dot(&input_array);
        
        let z_with_bias = z + &self.biases;

        // Apply activation (sigmoid here) element-wise and collect to Vec<f64>
        z_with_bias.mapv(|x| sigmoid(x))
        // z_with_bias.to_vec()
    }

    
}

fn mean_squared_error(predicted_val: &[f64], actual_val: &[f64]) -> f64 {
    predicted_val.iter().zip(actual_val.iter())
        .map(|(x, y)| (x-y).powi(2))
        .sum::<f64>() / (predicted_val.len() as f64)
}


fn layer_test() {
    let layer = Layer::new(10000,100);
    let input = vec![0.5;10000];

    let start_time = Instant::now();
    let forward_result = layer.forward(&input);
    let forward_duration = start_time.elapsed();
    println!("Time taken for forward(): {:?}", forward_duration);

    // dont need this anymore now that we are going with ndarray
    // let start_time = Instant::now();
    // let forward_result = layer.ndarray_forward(&input);
    // let forward_duration = start_time.elapsed();
    // println!("Time taken for ndforward(): {:?}", forward_duration);
}

fn nn_test() {
    let nn = NeuralNet::new(&[3, 2, 2,5]);
    let input = vec![0.5;3];
    
    let learning_rate = 0.1;
    let target = [5.0,1.0,3.0,2.0,4.0];

    nn.cycle(&input, &target, learning_rate);
    // for x in nn.layers {
    //     println!("{:?} \n",x)
        
    // }

    // let start_time = Instant::now();
    // let forward_result = nn.nd_forward(&input);
    // let forward_duration = start_time.elapsed();
    // // println!("{:?}",nn);
    // println!("Time taken for forward(): {:?}", forward_duration);

}
fn main() {
    nn_test();
}