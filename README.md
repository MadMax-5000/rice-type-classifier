# Rice Type Classification using Neural Networks

A PyTorch-based binary classification model to identify rice types based on morphological features.

## 📊 Dataset

This project uses the [Rice Type Classification Dataset](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification) from Kaggle, which contains morphological measurements of rice grains.

### Features:
- Area
- Major Axis Length
- Minor Axis Length
- Eccentricity
- Convex Area
- Equivalent Diameter
- Extent
- Perimeter
- Roundness
- Aspect Ratio

## 🚀 Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/MadMax-5000/rice-type-classifier.git
cd rice-type-classifier
```

2. Install required packages:
```bash
pip install torch torchvision torchsummary scikit-learn matplotlib pandas numpy opendatasets
```

3. Download the dataset:
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/mssmartypants/rice-type-classification")
```

> **Note**  
> You will need a Kaggle account and API credentials to download the dataset.

## 🏗️ Model Architecture

The project implements a simple feedforward neural network called "Astra" with:
- Input layer: 10 features
- Hidden layer: 20 neurons (can be modified)
- Output layer: 1 neuron (sigmoid activation for binary classification)
- Loss function: Binary Cross-Entropy (BCE)
- Optimizer: Adam (learning rate: 0.001)

## 📈 Training

The model is trained with:
- **Train/Validation/Test split**: 70/15/15
- **Batch size**: 64
- **Epochs**: 10
- **Normalization**: Min-max scaling (division by absolute maximum)

### Data Split:
- Training: 70%
- Validation: 15%
- Testing: 15%

## 🎯 Usage

### Training the Model

Run the main script to train the model:
```bash
python rice_classifier.py
```

The script will:
1. Load and preprocess the data
2. Train the model for 10 epochs
3. Display training and validation metrics
4. Generate loss and accuracy plots
5. Evaluate on the test set

### Making Predictions

After training, the model prompts for manual input of rice grain features:

```python
Area: 15000
Major Axis Length: 300
Minor Axis Length: 150
Eccentricity: 0.85
Convex Area: 15100
EquivDiameter: 138
Extent: 0.75
Perimeter: 800
Roundness: 0.9
AspectRation: 2.0
```

The model will output the predicted class (0 or 1).

## 📊 Results

The model generates two plots:
1. **Training and Validation Loss** over epochs
2. **Training and Validation Accuracy** over epochs

Expected performance: ~98%+ accuracy on the test set (varies with random seed).

## 🛠️ Project Structure

```
rice-type-classifier/
├── rice_classifier.py          # Main training script
├── riceClassification.csv      # Dataset (after download)
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## 📝 Requirements

Create a `requirements.txt` file with:
```
torch>=1.9.0
torchsummary>=1.5.1
scikit-learn>=0.24.2
matplotlib>=3.3.4
pandas>=1.2.4
numpy>=1.19.5
opendatasets>=0.1.22
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset provided by [mssmartypants](https://www.kaggle.com/mssmartypants) on Kaggle
- Built with PyTorch framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

> **Note**
> Make sure to update the file path in the code from `/content/rice-type-classification/riceClassification.csv` to match your local directory structure.
