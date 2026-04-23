# Dataset Information

This project uses the following datasets:
- **ShareGPT**: Primary dataset for fine-tuning Medusa heads.
- **Chatbot Arena Conversations**: Optional alternative/supplement.

## Automation
The logic for downloading and preprocessing these datasets is contained in `final/code/data_utils.py`. 

To download the data, run the utility script (implementation pending):
```bash
python code/data_utils.py --download
```

Data will be stored in this directory (`final/data/`) and is ignored by git to avoid committing large files.
