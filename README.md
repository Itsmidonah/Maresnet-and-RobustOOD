# 🎯 **Out-of-Distribution Detection in Transcription Factor Binding Site Prediction**

> **A case study of MAResNet in RobustOOD**

This thesis investigates the enhancement of out-of-distribution (OOD) generalization in deep learning models for **Transcription Factor Binding Site (TFBS)** prediction. It centers on **MAResNet**, a state-of-the-art multi-scale attention residual network, and evaluates its robustness using both quantitative and qualitative OOD settings.

---

## 🧠 Project Highlights

* 🔬 **TFBS Prediction with MAResNet**
* 🔁 **Full Reproduction and Training Pipeline**
* 📉 **Quantitative OOD Detection (e.g., MSP)**
* 🚀 **Integration into RobustOOD Framework**
* 🧪 **Qualitative OOD Evaluation**

---

## 🧪 **MAResNet Reproduction Guide**

Reproducing the original **MAResNet** implementation and adapting it for OOD detection tasks.

---

### ⚙️ Step 1: Environment Setup

1. **🔗 Connect to Remote GPU Machine**

   * Use OpenVPN for performance.
   * Ensure admin rights for GPU usage.

2. **🐍 Python Environment**

   ```bash
   conda create -n py38 python=3.8
   conda activate py38
   git clone <maresnet_repo_url>
   pip install -r requirements.txt
   ```

   ✅ Required packages:

   * `torch==1.8.1`
   * `torchvision==0.9.1`
   * `numpy==1.20.2`
   * `pandas==1.2.3`
   * `scikit_learn==0.24.2`

   ❗ *Ensure compatibility with your CUDA version.*

---

### 📁 Step 2: Dataset Setup

1. **🌐 Download Datasets**
   🔗 [MAResNet Datasets](https://csbioinformatics.njust.edu.cn/maresnet/)

2. **🚚 Upload to Server**

   ```bash
   scp /path/to/dataset.zip user@remote:/remote/path/
   ```

3. **📝 Rename Dataset Folder**

   ```bash
   mv TransferDataSet 690_dataset
   ```

4. **📂 Organize Global Dataset**

   ```bash
   mkdir -p Dataset/global_dataset
   mv global_train.data Dataset/global_dataset/train.data
   mv global_valid.data Dataset/global_dataset/valid.data
   mv global_test.data Dataset/global_dataset/test.data
   ```

5. **🗂️ Unzip**

   ```bash
   unzip dataset.zip -d ~/tfbsed/maresnet/Dataset/
   ```

6. ✅ **Ensure Directory Structure**

   ```
   Dataset/
   ├── 690_dataset/
   ├── cell_dataset/
   └── global_dataset/
   ```

---

### 🧪 Step 3: Create `valid.data` (if missing)

Use `create_valid_data.py` to split `train.data` (80:20 rule).
Add a seed for reproducibility.

```bash
python create_valid_data.py
```

---

### 🏋️ Step 4: Train on 690 ChIP-seq Dataset

```bash
python transform_on_690datasets.py
```

📂 Outputs:

* `checkpoint_transfer/` → models
* `runs_transfer/` → logs and results

---

### 🧬 Step 5: Train on Cell Datasets

Ensure folders contain `train.data`, `valid.data`, `test.data`.

```bash
python train_on_cell_datasets.py
```

📂 Outputs:

* `checkpoint/` → models
* `runs/` → logs

---

### 🌍 Step 6: Train on Global Dataset

⚠️ **Format issues** encountered. Ensure `.data` files contain:

* Unique ID
* Sequence
* Label

```bash
python train_on_global_dataset.py
```

---

### 📈 Step 7: View Results

* Models:

  ```bash
  ls checkpoint_transfer/
  ```

* Logs & Predictions:

  ```bash
  ls runs_transfer/
  ```

📌 *Use your custom metric scripts to evaluate model performance.*

---

### 🛠️ Troubleshooting

* ❗ *Check CUDA version:*

  ```bash
  nvidia-smi
  ```

* 🔗 [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

* ⚠️ No pretrained model provided – training is from scratch.

---

## 📦 MAResNet Dataset Structure

```bash
Dataset/
├── 690_dataset/
│   ├── wgEncodeAwgTfbsBroadDnd41CtcfUniPk/
│   │   ├── train.data
│   │   ├── valid.data
│   │   └── test.data
├── cell_dataset/
│   ├── A549/
│   └── ...
└── global_dataset/
    ├── train.data
    ├── valid.data
    └── test.data
```

---

## 🧠 **Quantitative OOD Detection Using MSP**

After MAResNet is successfully trained on the 690\_dataset, we used **Maximum Softmax Probability (MSP)** to evaluate its performance on **quantitative OOD** settings.

➡️ *We execute softmax in MAResNet train_test_api/test_api.py script, train again and make classwise plots, and shuffle the data (see custom shuffling script) and then train again. Evaluate their Maximum softmax Probability with classwise plots.*

---

# 🔐 **Reproducing RobustOOD Framework**

Paper:
📄 *"Towards more robust transcription factor binding site classifiers using out-of-distribution data (ICAART 2025)"*
**Authors:** István Megyeri & Gergely Pap

---

### ⚙️ Environment Setup

```bash
conda create -n tfood python=3.11.9
conda activate tfood
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas==2.2.1 tqdm==4.66.4 matplotlib==3.8.4 scikit-learn==1.4.2
pip install tensorboard==2.16.2
```

---

### 🏃 Model Training

```bash
./train3tf.sh
```

📊 View Training Logs:

```bash
tensorboard --logdir ./saved_models_wrn_cyclic
```

---

### 📊 Evaluate Trained Models

```bash
./eval3tf.sh
```

📉 Generate Result Figures:

```bash
python make_fig.py --fnames results_2024-09-04_wrn_cyclic_3tf_test.csv
```

---

## 🔁 **Inserting MAResNet into RobustOOD Framework**

We inserted the reproduced MAResNet model into the `TF_OOD_Robustness` pipeline by modifying the model registry and integrating the new architecture for qualitative OOD detection.

---

## 🧪 **Qualitative OOD Evaluation**

We tested MAResNet inside **RobustOOD** using diverse qualitative OOD datasets (e.g., different TFs, unseen experimental conditions).

🎯 Evaluation Goals:

* Generalize beyond training TF classes
* Detect unseen biological contexts
* Compare against baseline WRN models

📈 Results demonstrate MAResNet’s capability to outperform standard architectures in qualitative OOD detection.

---

**Source Code Extracts:**

Download and Unzip the folders inside here  https://drive.google.com/file/d/11YphTmYAgVQXg6KZfbxJoYQ7cdEvrjJI/view?usp=drive_link
with Title Source codes_Dorothy_Nabakooza_NY5O03.zip
Then follow this Readme for Implementation.

**Note** It is vital to note that these are 2 projects, you will need to clone twice and run each independently. When testing OOD impact of MAResNet in RobustOOD, use the RobustOOD implementation. 

## 📬 Contact

📧 **Author:** [Dorothy Nabakooza](mailto:nabakooza.dorothy@gmail.com)
🎓 MSc in Computer Science, University of Szeged
💬 Reproduction + OOD Integration Contributor
- C/O
-  Supervisor [István Megyeri](mailto:imegyeri@inf.u-szeged.hu)
- **RobustOOD Authors:** [István Megyeri](mailto:imegyeri@inf.u-szeged.hu), [Gergely Pap](mailto:papgergely93@gmail.com), Yahan
[István Megyeri](mailto:imegyeri@inf.u-szeged.hu), [Gergely Pap](mailto:papgergely93@gmail.com), Yahan
- **MAResNet Authors:** [Long-Chen Shen](mailto:shenlc1995@gmail.com), [Dong-Jun Yu](mailto:njyudj@njust.edu.cn)


---
