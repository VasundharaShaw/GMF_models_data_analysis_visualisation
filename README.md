# GMF_models_data_analysis_visualisation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Tools to **compare Galactic Magnetic Field (GMF) models** with **C-BASS 5 GHz polarization** data.  
Includes end-to-end pipelines for loading observational maps, harmonizing synchrotron templates, masking, fitting, and generating comparison plots.

📄 **Related material**
- [DESY 2025 GMF–C-BASS talk (slides)](https://drive.google.com/file/d/15HRuauIdqdiJdtrGDF_ruxOOCKgCX-uO/view)  
- *Paper in preparation:* Shaw et al., Fall 2025
- *Contact V. Shaw for relevant details at vasundhara.shaw@manchester.ac.uk as the work is still in progress*
---

## 🚀 Features

- Load & process polarization maps (C-BASS, S-PASS, Franken)
- Compute PI from Q/U with variance handling and RM masking
- Load & scale GMF synchrotron templates: **JF12, UF23, SVT22, KST24, XH19, LogSpiral**
- Frequency scaling across bands (e.g. **30 GHz → 4.76 GHz**)
- Apply Galactic masks (GC, quadrants, N/S, high-latitude)
- Fit models via **amplitude scaling** and **Spearman correlation**
- Generate **heatmaps**, **T–T plots**, and **region-wise tables**

---

## 📦 Requirements

This project requires **Python 3.10+** and the following libraries:

| Package       | Recommended Version |
|---------------|----------------------|
| numpy         | 1.23+               |
| scipy         | 1.10+               |
| pandas        | 2.0+                |
| healpy        | 1.16+               |
| matplotlib    | 3.7+                |
| seaborn       | 0.12+               |
| astropy       | 5.3+                |
| scikit-image  | 0.21+               |
| cmcrameri     | 1.7+                |
| colorcet      | 3.0+                |

> ⚠️ Ensure your `numpy`/`scipy` versions are compatible with **healpy** and **astropy**.  
> Using a virtual environment (`conda` or `venv`) is **highly recommended**.

---

## ⚙️ Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/<your-username>/GMF_models_data_analysis_visualisation.git
cd GMF_models_data_analysis_visualisation

conda create -n gmf python=3.10
conda activate gmf

pip install numpy==1.23.5 scipy==1.10 pandas==2.0 healpy==1.16.5 matplotlib==3.7 \
            seaborn==0.12 astropy==5.3 scikit-image==0.21 cmcrameri==1.7 colorcet==3.0


## 📊 Example Outputs

- **Polarized intensity maps**: C-BASS/S-PASS data vs GMF model templates  
- **Heatmaps**: best-fit amplitudes & Spearman correlation across sky regions  
- **T–T scatter plots**: model vs. data comparisons in quadrants, GC, N/S, high-latitude masks  
- **Tables**: CSVs of fitted amplitudes, correlation coefficients, and regional statistics  

---

## 📚 References

### GMF Models
- **SVT22** – Shaw et al., *Synchrotron emission in Galactic Magnetic Fields*, MNRAS 517, 2534 (2023)  
- **UF23** – Unger & Farrar (2023), *arXiv:2311.12120*  
- **KST24** – Kim, Seta & Thomson (2024), *arXiv:2407.02148*  
- **JF12** – Jansson & Farrar (2012), *ApJ 757, 14*  
- **XH19** – Han et al. (2019), *MNRAS 486, 4275*  
- **LogSpiral** – Page et al. (2007), *ApJ 665, 1067*  

### Observational Data
- **C-BASS** – C-Band All-Sky Survey  
- **S-PASS** – Carretti et al. 2019, *MNRAS 484, 4933*  

---

## 📜 Citation

If you use this repository, please cite:

- Shaw et al., *C-BASS collaboration paper* (in preparation, Fall 2025).  
- [DESY 2025 GMF–C-BASS talk (slides)](https://drive.google.com/file/d/15HRuauIdqdiJdtrGDF_ruxOOCKgCX-uO/view).  

And, where appropriate, cite the GMF model papers listed in the [References](#-references) section.
