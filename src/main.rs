use rand::Rng;
use neural_net::utils::{self, sigmoid};
use ndarray::{Array1, Array2, ArrayView1};
use std::time::Instant;


#[derive(Debug)]
struct NeuralNet {
    layers: Vec<Layer>
}

impl NeuralNet {
    fn new(layer_sizes: &[usize]) -> NeuralNet {
        // iterating over the layer array input by window of 2 we create new layers
        let layers = layer_sizes.windows(2)
            .map(|w| Layer::new(w[0],w[1]))
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
            activations.push(activation.clone());
            activation
            
        });

        activations
    }

    fn cycle(&self, input: &[f64], target: &[f64], learning_rate: f64) -> () {
        
        // get activations from forward pass
        let activations = self.forward(input);
        let output = activations.last().expect("Last vector was empty");

        // compute the deltas from final activation
        let error = output.iter().zip(target.to_vec()).map(|(a, y)| a - y);

        self.layers.iter().rev().zip(activations).map(|(layer, activation)| {
            // calculate gradient
            let z = 
            let gradient
        })
    }

    fn nd_forward(&self, input: &[f64]) -> Vec<f64> {
        self.layers.iter().fold(input.to_vec(), |acc, layer| layer.ndarray_forward(&acc))

    }
}

#[derive(Debug)]
struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Layer {
        let mut rng = rand::thread_rng();

        //Initialize the weights and biases randomly
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();

        let biases = (0..output_size)
            .map(|_| rng.gen_range(-1.0..1.0)).collect();

        Layer { weights, biases}
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights.iter().enumerate().map(|(i, neuron_weights)| {

            // iterating over each neuron, multiply the weight & the input value and add the bias
            // then apply the sigmoid function to flatten TODO: can make this a parameterized call based on sigmoid or ReLu later
            let sum: f64 = neuron_weights.iter().zip(input.iter())
                .map(|(w,i)| w * i)
                .sum();
            sigmoid(sum + self.biases[i])
        }).collect()    
    }

    fn backward(&self, ) -> () {
        // 
        
    }


    fn ndarray_forward(&self, input: &[f64]) -> Vec<f64> {
        let output_size = self.weights.len();
        let input_size = self.weights[0].len();   

        // flatten the weights from array into a vector
        let flattened_vec = self.weights.iter()
            .flat_map(|n| n.iter().copied())
            .collect();
        
        // Create ndarray arrays
        let weights_array = Array2::from_shape_vec((output_size, input_size), flattened_vec)
            .expect("Failed to create weights array");

        // Interestingly... this can be done without copying using ArrayView1 feature from ndarray, which could be faster
        let biases_array = ArrayView1::from(&self.biases);
        let input_array = ArrayView1::from(input);

        // Or done with copying, which increases memory OH
        // let biases_array = Array1::from(self.biases.to_vec());
        // let input_array = Array1::from(input.to_vec());


        // Perform matrix-vector multiplication and add biases
        let z = weights_array.dot(&input_array);

        let z_with_bias = z + biases_array;

        // Apply activation (sigmoid here) element-wise and collect to Vec<f64>
        z_with_bias.mapv(|x| sigmoid(x));
        z_with_bias.to_vec()
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


    let start_time = Instant::now();
    let forward_result = layer.ndarray_forward(&input);
    let forward_duration = start_time.elapsed();
    println!("Time taken for ndforward(): {:?}", forward_duration);
}

fn nn_test() {
    let nn = NeuralNet::new(&[3, 2, 2,5]);
    let input = vec![0.5;1000];
    
    for x in nn.layers {
        println!("{:?} \n",x)
        
    }

    // let start_time = Instant::now();
    // let forward_result = nn.nd_forward(&input);
    // let forward_duration = start_time.elapsed();
    // // println!("{:?}",nn);
    // println!("Time taken for forward(): {:?}", forward_duration);

}
fn main() {

}