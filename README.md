# Machine-Learning-Classification-Models-using-NBA-Dataset

# NBA Player Position Prediction

## Overview

This Python script is designed to predict the position of NBA players based on their performance statistics. It uses machine learning models such as Linear Support Vector Classifier (Linear SVC) and includes data preprocessing steps to filter out players with low playing time.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- scikit-learn

You can install the dependencies using the following command:

```bash
pip install numpy pandas scikit-learn
```

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/nba-player-prediction.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd nba-player-prediction
    ```

3. **Run the script:**

    ```bash
    python nba_prediction_script.py
    ```

    Note: Make sure you have the `nba2021.csv` file in the same directory as the script.

## Configuration

You can customize the script by modifying the following parameters:

- `train_size`: The proportion of the dataset used for training (default: 0.75).
- `test_size`: The proportion of the dataset used for testing (default: 0.25).
- Model-specific parameters (e.g., `n_neighbors` for KNeighborsClassifier, `max_depth` for DecisionTreeClassifier, etc.).

## Results

The script will output the accuracy of the model on the test set, confusion matrix, and cross-validation scores. These results provide insights into the performance and generalization ability of the model.

## Author

- Aravindh Gopalsamy
- gopal98aravindh@gmail.com

## License

This project is not open for external use or distribution. All rights reserved.

## Important Note for Students

**Warning:** This code is intended for educational purposes only. Please do not use this code for any assignment, and consider it as a reference implementation. Use your own implementation for academic assignments.

## Acknowledgments

- Data source: [nba2021.csv](link-to-data-source)

---

Feel free to add or modify sections based on your specific needs. Include any additional information that you think would be helpful for users or contributors.
