# NIST/TRC Experimental Data

This directory contains experimental binary liquid–liquid equilibrium (LLE) data obtained from the **NIST Thermodynamics Research Center (TRC)**, as used in the associated publication.  

The dataset originates from the [PGL database](https://pgl6ed.byu.edu/) and is distributed with permission via the [PGLWrapper repository](https://github.com/PGLadmin/PGLWrapper).  

---

## Files in this directory

- **`LLeDbPGL6ed96b.txt`**  
  Original LLE database file as provided by NIST/TRC (downloaded from [PGLWrapper](https://github.com/PGLadmin/PGLWrapper)).  

- **`LLeDbPGL6ed96b.py`**  
  Helper script to parse the `.txt` file and convert it into a structured format (`.csv`).  

- **`LLeDbPGL6ed96b.csv`**  
  Direct CSV export of the raw `.txt` file for easier handling.  

- **`LLeDbPGL6ed96b.xlsx`**  
  Final standardized dataset in tabular form. This version has been cleaned and formatted for consistency (e.g., harmonized units, unified schema).  

- **`LLeDbPGL6ed96b.pdf`**  
  Multi-panel plot of all 92 binary systems, showing experimental LLE data together with COSMO-SAC model results.  

---

## Notes

- The `.xlsx` file is the recommended entry point for working with the experimental data in a structured and standardized form.  
- Calculated model results are available separately under `data/calculated/lle_results` (`lle-SAC_2010.csv` and `lle-SAC_dsp.csv`).  
- Both raw and processed experimental data are provided to ensure transparency and reproducibility.  
