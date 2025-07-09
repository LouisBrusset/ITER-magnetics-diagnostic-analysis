# ITER-magnetics-diagnostic-analysis
Final project that I have done for ITER Organization (IO) during my 6-month internship.





# ITER-pendulum-physics-autoencoder
Try to recover the physics of an oscillating pendulum (ideal then amortized, simple then double) thanks to the embedded space of an Auto-Encoder.

## Description

This project contains a number of Jupyter notebooks designed to introduce different self-supervised machine learning architecture to recover the motion of a pendulum. Using synthetized data from ODE simulation in this repositories too, we will be able to recover physics parameters of the governing equations.

## Data Source

The data comes from non linear pendulum simulation.

## Getting Started

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) - A faster and more reliable Python package installer and resolver

### Installation

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/ITER-pendulum-physics-autoencoder.git
   ```

2. Navigate to the project directory

   ```bash
   cd ITER-pendulum-physics-autoencoder
   ```

3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) if you don't have it already

   ```bash
   pipx install uv
   ```

4. Create a virtual environment and install dependencies using uv

   ```bash
   uv venv
   uv pip install -e .
   ```

5. Activate the virtual environment

   ```bash
   # On Windows
   .venv\Scripts\activate

   # On Unix or MacOS
   source .venv/bin/activate
   ```

## Usage

### Running Jupyter Notebooks

To run the Jupyter notebooks, make sure you've activated your virtual environment, then:

```bash
# In VSCode
code .
# in your browser
jupyter lab --notebook-dir notebooks/
```

This will open a browser window with the Jupyter interface where you can select and run any of the notebooks.

### Available Notebooks

1. **test-data-creation-simple-non-amortized** - see name.
2. **test-data-creation-simple-non-amortized for hnn** - see name.
3. **test-data-creation-simple-amortized** - see name.
4. **test-data-creation-double-non-amortized** - see name.
5. **test-data-creation-double-amortized** - see name.

6. **test-sinus** - Test some functions to dertermine the pulsation of a cyclique motion in the simpliest way.
7. **test-pi-ae** - Physics-informed Autoencoder.
8. **test-latent-ode** - Autoencoder with an ODE simulation in the laten space.
9. **test-latent-ode-2** - Idem.
10. **test-invertible-ae** - normalizing flow technics with volume conservation.
11. **test-vae** - Variational Autoencoder.
12. **test-hamiltonian-nn** - Hamiltonian Neural Network.
13. **test-hamiltonian-nn 2** - Idem.
14. **test-hamiltonian-nn 3** - Idem.


## License

Not any licence for this project.
