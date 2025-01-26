# Adaptive Vision Transformers for Efficient Processing of Video Data in Automotive Applications

## Code

This repository contains code to run the model designed in 'Adaptive Vision Transformers for Efficient Processing of Video Data in Automotive Applications'. Below, you'll find details about the code structure, how to set up the environment, run inferences, and interpret benchmarking results.

### Model

[model](./code/model) contains the primary implementation of the model, extending the mmengine and mmsegmentation frameworks.

* [encoder-decoder](./code/model/segmenter/encoder_decoder.py): Modified encoder-decoder implementation.
* [token reducing vision transformer](./code/model/backbone/token_reducing_vit.py): Token-reducing Vision Transformer module.

### Setup and execution

* [setup](./code/setup.ipynb): Notebook to install model weights and dependencies.

### Running the code
* [example.ipynb](./code/notebooks/example.ipynb): Example notebook demonstrating inference with the model.
* [benchamrking](./code/notebooks/benchmarking.ipynb): Benchmarking and analysis

### Benchmark results

[benchamrking](./code/notebooks/benchmarking) contains numpy files with benchmarking results.

#### File descriptions

* starting with encode_times: Measures the time taken to run the encoder. (in seconds)
* starting with pixel_wise_acc: Shows the pixel-wise accuracy loss compared to the original model. (in %)
* starting with reduced_tokens_heatmap: Visualizes the location of the most pruned tokens.
* starting with pruned_tokens: Represents the amount of tokens that are most pruned. (absolute numbers)

#### File naming conventions
* Files ending with '0.xx': Fixed threshold with a standard reduction interval of 8.
* Files containing 'lin': Linear threshold.
* Files containing 'all_layers': Reduction interval set to 1.
* Files containing 'int_x': Representing varying reduction intervals.
