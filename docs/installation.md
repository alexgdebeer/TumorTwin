# ⚙️ Installation Guide for TumorTwin

TumorTwin is tested with **Python 3.11**, and we recommend using a virtual environment for a clean setup.

## Prerequisites

- Python (tested on Python **3.11**)
- `git`
- A package manager like [Anaconda](https://docs.anaconda.com/anaconda/install/) or `pip`

---

## Option 1: Using Anaconda (recommended for most users)

1. **Install Anaconda** (if not already installed):  
   https://docs.anaconda.com/anaconda/install

2. **Create a new environment with Python 3.11**:

   ```bash
   conda create -n tumortwin python=3.11 anaconda
   ```

3. **Activate the environment**:

   ```bash
   conda activate tumortwin
   ```

---

## Option 2: Using `venv` (lightweight setup)

1. **Create a virtual environment**:

   ```bash
   python3.11 -m venv tumortwin-env
   ```

2. **Activate it**:

   - On macOS/Linux:

     ```bash
     source tumortwin-env/bin/activate
     ```

   - On Windows:

     ```powershell
     .\tumortwin-env\Scripts\activate
     ```

---

## Install TumorTwin

1. **Clone the repository**:

   ```bash
   git clone https://github.com/OncologyModelingGroup/TumorTwin.git
   cd TumorTwin
   ```

2. **Install dependencies**:

   - Standard installation:

     ```bash
     pip install .
     ```

   - Development mode (editable install + dev tools):

     ```bash
     pip install -e ".[dev]"
     ```

---

## Optional: Running the Demo

   Open `HGG_Demo.ipynb` or `TNBC_Demo.ipynb` in Jupyter, VS Code, Colab or your preferred IDE.

---