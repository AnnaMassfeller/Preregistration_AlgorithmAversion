# Preregistration_AlgorithmAversion

In this repository the code that was used to create the synthetic data and rund the models as described in the pre-registration is stored:

- requirments: required versions of packages needed to run the code properly
- Preregistration.py: Python file to be run first, in here the synthetic data is created that will be used (also in th R code).
    The file is organized in 3 steps:   
    1. Defining the model
    2. Creating synthetic data + prior predicitve checks
    3. Running the MCMC + posterior predicitve checks
- SyntehticData: folder in which the synthetic data from "Preregistration.py" is stored (interim)
- Preregistration.R: R file to be run second. Uses the synthetic data created in "Preregistration.py" and stored in "SyntheticData". Used to run "traditional approaches" as comparison to probabilistic approach.
