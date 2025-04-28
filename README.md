# BOLT

BOst Learner Transpiler


## About

BOLT is a script for transpiling a XGBRegressor serialized model into a C header.
This header provides a function to natively run the model inference.
The function arguments are the input features of the model, while the function return is the target variable learned by the model.
```
float example(int feature1, float feature2, bool feature3)
{
    ...
}
```

XGBRegressor trains multiple decision trees that build upon previous trees errors.
Each decision tree learned by XGBRegressor turns into nested if-else statements that traverse the tree nodes.
Each if-else statement provides a tree split, that compares one input feature to one learned condition.
When all nodes of a tree are traversed, it arrives at a leaf, that contains the tree weight.
```
  if (feature1 < condition1) {
    if (feature2 < condition2) {
        w0 = 7.0;
    } else {
        ...
    }
  } else {
    ...
  }
```

Each tree individually result in a weight, and the final prediction is the sum of all tree weights with a bias.
```
return 0.5 + w0 + w1 + ... + wn;
```

## Build and Install

### Development build

It is recommended to run inside a python venv.
After setting up the venv, install as an editable package by running from this project folder:

```
pip install -e .
```

### Release build

Install directly to the machine (requires python-build):

```
$ python -m build
# python -m installer dist/*.whl
```


## Usage

### Basic usage

Export the XGBRegressor model using `save_model` as a json file.
Run BOLT:
```
bolt input.json
```
This will create an `input.h` file with a `float input(args...)` function based on the model input.

It is possible to select the output file and output function names:
```
bolt input.json -o model.h -f function
```
This will create an `model.h` file with a `float function(args...)` function based on the model input.

### Collapsing dummies

When a model has one-hot categorical data, XGBRegressor represents internally with _indicator_ type.
This means that each possible category of a variable will turn into a single boolean variable.
E.g., a `category` variable turns into `category0`, `category1`, `category2` and so on.
A BOLT-ed model will result in the following:
```
float function(bool category0, bool category1, ...)
{
  if (category0 == true) {
    if (category1 == true) {
      ...
    }
  } else {
    ...
  }
}
```

When transpiling the model to native C, _indicator_ type does not play well, as each category will result in an individual argument to the model function.
BOLT provides an option to collapse dummy variables into a single integer value-encoded variable.
E.G., the `categoryN` dummy variables will turn into a single `category` integer variable, with internal comparisons matching its dummy label (0 for `category0`, 1 for `category1` and so on).
A BOLT-ed model with collapsed dummies will result in the following:
```
float function(int category)
{
    if (category == 0) {
        if (category == 1) {
            ...
        }
    } else {
        ...
    }
}
```

To enable dummy variable collapsing, run:
```
bolt input.json -c
```

### Leaf quantization

Traversing a tree is fast when its splits are integer comparison, as each split will potentially result in a branch instruction.
In this case, most of the processing time happens in the final accumulation of the weights.
In a CPU without FPU, the floating-point accumulation takes approximately 75% of function time.

BOLT provides an option to quantize the tree weights to integer, by multiplying all weights by a power-of-2 value.
This turns the accumulation into several integer additions, followed by a right bit shift by the log2 of the quantization multiplier.
```
return (128 + w0 + w1 + ... + wn) >> 8;
```

To enable leaf quantization, run:
```
bolt input.json -q 256
```
Where the value after the `-q` switch is a power-of-2 value to multiply every weight.

### Integer minimization

Internally XGBoost represents everything as 'float', 'int', or 'int64_t'.
In C, we may want to represent some feature inputs as smaller integers, such as uint8_t, uint16_t, int8_t and int16_t.
The integer minimization options scans all values used by integers and changes the feature inputs to the miminium size.

To enable integer minimization, run:
```
bolt input.json -m
```

### Linear feature quantization

XGBoost performs comparison to trained feature weights.
The range of values of a comparison is, therefore, defined at training time.
It is possible to decrease the magnitude of a weight by subtracting an offset.
This can potentially decrease the feature size.

To enable linear quantization, run:
```
bolt input.json -l
```
