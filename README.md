# Credit Card Fraud Detection Analysis

## 📊 Dataset Setup Instructions

This notebook has been configured to work with **manually downloaded datasets**. Follow these simple steps:

### Quick Setup (Option 1 - Recommended)

1. **Download the dataset**:
   - Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Download the `creditcard.csv` file
   
2. **Place the file**:
   - Copy `creditcard.csv` directly into this project folder:
   - `c:\Users\HP\University\Evaluating-Fraud-Detection\`

3. **Run the notebook**:
   - The system will automatically detect the dataset
   - Proceed with the analysis

### Organized Setup (Option 2)

1. Create a `data` folder in the project directory
2. Place `creditcard.csv` inside the `data` folder
3. The system will automatically find it

### Supported File Names

The system recognizes these filename variations:
- `creditcard.csv` (recommended)
- `credit_card_fraud.csv`
- `fraud_detection.csv`
- `creditcardfraud.csv`
- `Credit_Card_Fraud_Detection.csv`

### File Verification

Expected dataset specifications:
- **Size**: ~144 MB
- **Columns**: 31 (V1-V28, Time, Amount, Class)
- **Rows**: ~284,807 transactions
- **Format**: CSV

## 🚀 Running the Analysis

1. **Execute Cell 1**: Import libraries
2. **Execute Cell 2**: Configure dataset paths (auto-detection)
3. **Execute Cell 3**: Load dataset
4. **Continue**: Follow the notebook cells sequentially

## 📁 Expected Outputs

After running all cells:
- 5 publication-ready PNG visualizations
- 2 CSV files with performance metrics
- Complete academic analysis ready for submission

## ❓ Troubleshooting

**Dataset not found?**
- Ensure the CSV file is in the project directory
- Check that the filename matches one of the supported names
- Verify the file is not corrupted (should be ~144 MB)

**Missing dependencies?**
- Run: `pip install -r requirements.txt` (if available)
- Or install manually: `pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn imbalanced-learn`


---

**Note**: If no dataset is found, the notebook will generate synthetic data for demonstration purposes.
