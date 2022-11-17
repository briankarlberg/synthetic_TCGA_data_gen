# synthetic_TCGA_data_gen  
Generative deep learning model development for synthetic TCGA omic sample generation  

2022-11-12
EDA in non-RNA
    result: methylation data is continuous like m(i)RNA
    CNVR and MUT can be one-hot-encoded to benchmark against a transformer
Intersections of features across cohorts: a_ntrst_00

2022-10-25
pip install -r requirements.txt

2022-10-07  
First run float64 BLCA on 2D convolution VAE  
Next: MinMax norm, outlier clipping, log scaling     

2022-10-06  
Convert data type from UNIT8 to float in 2D conv MNIST model  
Build a float sequential model  
UMAPs complete by 2pm Friday  
collect code in synthetic_TCGA_data_gen  

----------
archive folder: BRCA UMAP is a plot of real vs synthetic gene expression samples. The synthetic samples in this development version were generated with the VAE set to a latent space dimension of 4.
