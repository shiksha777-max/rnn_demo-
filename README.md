# Vanilla RNN — Airline Passenger Forecasting

A beginner-friendly implementation of a **Simple Recurrent Neural Network (SimpleRNN)** built with TensorFlow/Keras to predict monthly airline passenger counts.

---

## Overview

This notebook walks through building a Vanilla RNN from scratch to forecast time-series data — specifically, the classic **airline passengers dataset**. Unlike a standard feedforward network, an RNN reads inputs **one step at a time** and maintains a **hidden state** (memory) that carries information forward across time steps.
Step 1: Read month-1  -> update hidden state
Step 2: Read month-2  -> update hidden state (remembers month-1)
Step 3: Read month-3  -> update hidden state (remembers months 1 & 2)
Output: Dense layer   -> single passenger count prediction
---

## Project Structure
vanilla_rnn_airline_passengers/
|
|-- vanilla_rnn_airline_passengers.ipynb
|-- README.md
|-- data/
---

## Getting Started

### Prerequisites

Python 3.8 or higher.

### Installation

```bash
git clone https://github.com/your-username/vanilla_rnn_airline_passengers.git
cd vanilla_rnn_airline_passengers
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### Running in Jupyter

```bash
jupyter notebook vanilla_rnn_airline_passengers.ipynb
```

> **Tip:** If you get a `ModuleNotFoundError` for TensorFlow inside Jupyter, run this in a notebook cell:
>
> ```python
> import sys
> !{sys.executable} -m pip install tensorflow
> ```
>
> Then restart the kernel and re-run.

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Building and training the RNN model |
| `scikit-learn` | Preprocessing and evaluation metrics |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting predictions vs actuals |

---

## Model Architecture

Input -> SimpleRNN -> Dense(1) -> Prediction
- **Input:** Sequence of past monthly passenger counts
- **SimpleRNN:** Learns temporal dependencies across time steps
- **Dense:** Outputs a single predicted passenger count
- **Optimizer:** Adam
- **Loss:** Mean Squared Error

---

## Dataset

The [Airline Passengers dataset](https://www.kaggle.com/datasets/chirag19/air-passengers) contains monthly totals of international airline passengers from **1949 to 1960** (144 data points).

---

## Evaluation

The model is evaluated using:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error

Predictions vs actual passenger counts are plotted at the end of the notebook.

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.

---

