########################################################################
###Use synthetic data for "tradiitonal approaches" in  pre-registration#
#######################################################################

#install packages
library(AER)
library(psych)
library(ggplot2)
library(readr)
library(seminr)
library(ggiraph)
library(ggiraphExtra)
library(jtools)
library(readr)
library(ggplot2)
library(RColorBrewer)
library(broom)
library(seminr) #used for PLS-SEM

#read data as created in python files
Data_H1True <- read_csv("synthetic_data_H1True.csv")
View(Data_H1True)

Data_NoH1 <- read_csv("synthetic_data_NoH1.csv")
View(Data_NoH1)

#######################################################################
###1.Run PLS-SEM with synthetic data for pre-registration (UTAUT part)#
#######################################################################

# Create measurement model
simple_mm <- constructs(
 composite("PE", multi_items("PE_", 1:4)),
 composite("EE", multi_items("EE_", 1:3)),
 composite("SI", multi_items("SI_", 1:2)),
 composite("AIA", multi_items("AIA_",1:16)),
 composite("BI", multi_items("BI_",1:3)))


# Create structural model
simple_sm <- relationships(
 #
 paths(from = c("SI","PE","EE", "AIA"), to = c("BI")),
 paths(from = c("AIA"), to = c("PE","EE")))# 


# Estimate the model
#first with synthetic data where H1 is true
simple_model_H1True <- estimate_pls(data = Data_H1True ,
                                    measurement_model = simple_mm,
                                    structural_model = simple_sm,
                                    inner_weights = path_weighting,
                                    missing = mean_replacement,
                                    missing_value = "-99")

# Summarize the model results
summary_simple_H1True <- summary(simple_model_H1True)
# Inspect the model’s path coeffcients and the R^2 values
summary_simple_H1True$paths
# Inspect the construct reliability metrics
summary_simple_H1True$reliability


# Bootstrap the model
boot_simple_H1True <- bootstrap_model(seminr_model = simple_model_H1True,
                                      nboot = 1000,
                                      cores = NULL,
                                      seed = 123)

# Store the summary of the bootstrapped model
sum_boot_simple_H1True <- summary(boot_simple_H1True)
# Inspect the bootstrapped structural paths
sum_boot_simple_H1True$bootstrapped_paths
# Inspect the bootstrapped indicator loadings
sum_boot_simple_H1True$bootstrapped_loadings

#plot the constructs’ internal consistency reliabilities (i.e., Cronbach’s alpha, rhoA, and rhoC)
plot(summary_simple_H1True$reliability)


plot(simple_model_H1True)
plot(boot_simple_H1True)

#################################################################
#then the same for data where H1 is not true
simple_model_NoH1 <- estimate_pls(data = Data_NoH1 ,
                                  measurement_model = simple_mm,
                                  structural_model = simple_sm,
                                  inner_weights = path_weighting,
                                  missing = mean_replacement,
                                  missing_value = "-99")

# Summarize the model results
summary_simple_NoH1 <- summary(simple_model_NoH1)
# Inspect the model’s path coeffcients and the R^2 values
summary_simple_NoH1$paths
# Inspect the construct reliability metrics
summary_simple_NoH1$reliability


# Bootstrap the model
boot_simple_NoH1 <- bootstrap_model(seminr_model = simple_model_NoH1,
                                    nboot = 1000,
                                    cores = NULL,
                                    seed = 123)

# Store the summary of the bootstrapped model
sum_boot_simple_NoH1 <- summary(boot_simple_NoH1)
# Inspect the bootstrapped structural paths
sum_boot_simple_NoH1$bootstrapped_paths
# Inspect the bootstrapped indicator loadings
sum_boot_simple_NoH1$bootstrapped_loadings

#plot the constructs’ internal consistency reliabilities (i.e., Cronbach’s alpha, rhoA, and rhoC)
plot(summary_simple_NoH1$reliability)


plot(simple_model_NoH1)
plot(boot_simple_NoH1)





##########################################################
###2. Run tobit model as comparison for experimental part#
##########################################################

#define tobit model
#first using data where H1 is true

#create construct of AI anxiety as mean of all statements
Data_H1True$AIAnxiety <- rowMeans(Data_H1True[,c("AIA_1","AIA_2","AIA_3","AIA_4","AIA_5","AIA_6","AIA_7","AIA_8","AIA_9","AIA_10","AIA_11","AIA_12","AIA_13","AIA_14","AIA_15","AIA_16")])

# Fit the Tobit model
tobit_model_H1true <- tobit(
 Delta_WTP_Q0_relTo90 ~ True_DeltaPerformance_Q0 + AIAnxiety,
 data = Data_H1True,
 left = -150,  # lower bound
 right = 150  # upper bound
)


# Summarize the model
summary(tobit_model_H1true)


#create plot similar to concept of python plots of prior and posterior predicitves
# Step 1: Create a sequence of True_DeltaPerformance_Q0 values to plot results
True_DeltaPerformance_seq <- seq(min(Data_H1True$True_DeltaPerformance_Q0),
                                 max(Data_H1True$True_DeltaPerformance_Q0), 
                                 length.out = 100)

# Step 2: Define AIAnxiety levels to plot
AIAnxiety_levels <- c(0, 1, 2, 3, 4, 5, 6)

# Step 3: Prepare the data for plotting
plot_data_H1true <- expand.grid(True_DeltaPerformance_Q0 = True_DeltaPerformance_seq,
                         AIAnxiety = AIAnxiety_levels)

# Predict Delta_WTP_Q0_relTo90 using the fitted Tobit model
plot_data_H1true$Delta_WTP_Q0_relTo90 <- predict(tobit_model_H1true, newdata = plot_data_H1true)

# Step 4: Plot the regression lines with reversed "Spectral" palette
reversed_palette <- rev(brewer.pal(n = length(AIAnxiety_levels), name = "Spectral"))

ggplot(plot_data_H1true, aes(x = True_DeltaPerformance_Q0, 
                      y = Delta_WTP_Q0_relTo90, 
                      color = as.factor(AIAnxiety))) +
 geom_line(linewidth = 1.2) +  # Add regression lines
 scale_color_manual(values = reversed_palette) +  # Use reversed Spectral palette
 labs(title = "Effect of True_DeltaPerformance on Delta_WTP by AI Anxiety-level when H1 is true",
      x = "True_DeltaPerformance",
      y = "Delta_WTP_relTo90",
      color = "Level of AIAnxiety") +
 ylim(-2, 2) +  # Set y-axis limits
 theme_minimal() +
 theme(legend.position = "bottom")


###
#then do the same using the data where H1 is not true
#create construct of AI anxiety as mean of all statements
Data_NoH1$AIAnxiety <- rowMeans(Data_NoH1[,c("AIA_1","AIA_2","AIA_3","AIA_4","AIA_5","AIA_6","AIA_7","AIA_8","AIA_9","AIA_10","AIA_11","AIA_12","AIA_13","AIA_14","AIA_15","AIA_16")])

# Fit the Tobit model
tobit_model_NoH1 <- tobit(
 Delta_WTP_Q0_relTo90 ~ True_DeltaPerformance_Q0 + AIAnxiety,
 data = Data_NoH1,
 left = -150,  # lower bound
 right = 150  # upper bound
)

# Summarize the model
summary(tobit_model_NoH1)


#plot similar to python
# Step 1: Create a sequence of True_DeltaPerformance_Q0 values to plot results
True_DeltaPerformance_seq <- seq(min(Data_NoH1$True_DeltaPerformance_Q0),
                                 max(Data_NoH1$True_DeltaPerformance_Q0), 
                                 length.out = 100)

# Step 2: Define AIAnxiety levels to plot
AIAnxiety_levels <- c(0, 1, 2, 3, 4, 5, 6)

# Step 3: Prepare the data for plotting
plot_data_NoH1 <- expand.grid(True_DeltaPerformance_Q0 = True_DeltaPerformance_seq,
                         AIAnxiety = AIAnxiety_levels)

# Predict Delta_WTP_Q0_relTo90 using the fitted Tobit model
plot_data_NoH1$Delta_WTP_Q0_relTo90 <- predict(tobit_model_NoH1, newdata = plot_data_NoH1)

# Step 4: Plot the regression lines with reversed "Spectral" palette
reversed_palette <- rev(brewer.pal(n = length(AIAnxiety_levels), name = "Spectral"))

ggplot(plot_data_NoH1, aes(x = True_DeltaPerformance_Q0, 
                      y = Delta_WTP_Q0_relTo90, 
                      color = as.factor(AIAnxiety))) +
 geom_line(linewidth = 1.2) +  # Add regression lines
 scale_color_manual(values = reversed_palette) +  # Use reversed Spectral palette
 labs(title = "Effect of True_DeltaPerformance on Delta_WTP by AI Anxiety-level when H1 is not true",
      x = "True_DeltaPerformance_Q0",
      y = "Delta_WTP_Q0_relTo90",
      color = "Level of AIAnxiety") +
 ylim(-2, 2) +  # Set y-axis limits
 theme_minimal() +
 theme(legend.position = "bottom")



#plot coefficients of both models in one plot:

# Extract coefficients for the model where H1 is true
coefficients_H1true <- summary(tobit_model_H1true)$coefficients
coefficients_df_H1true <- data.frame(
 term = rownames(coefficients_H1true),
 estimate = coefficients_H1true[, 1],  # Coefficient estimates
 std_error = coefficients_H1true[, 2],  # Standard errors
 conf.low = coefficients_H1true[, 1] - 1.96 * coefficients_H1true[, 2],  # 95% CI lower bound
 conf.high = coefficients_H1true[, 1] + 1.96 * coefficients_H1true[, 2],  # 95% CI upper bound
 model = "H1 True"  # Add a column to identify the model
)

# Extract coefficients for the model where H1 is not true
coefficients_NoH1 <- summary(tobit_model_NoH1)$coefficients
coefficients_df_NoH1 <- data.frame(
 term = rownames(coefficients_NoH1),
 estimate = coefficients_NoH1[, 1],  # Coefficient estimates
 std_error = coefficients_NoH1[, 2],  # Standard errors
 conf.low = coefficients_NoH1[, 1] - 1.96 * coefficients_NoH1[, 2],  # 95% CI lower bound
 conf.high = coefficients_NoH1[, 1] + 1.96 * coefficients_NoH1[, 2],  # 95% CI upper bound
 model = "No H1"  # Add a column to identify the model
)

# Combine both data frames
combined_coefficients_df <- rbind(coefficients_df_H1true, coefficients_df_NoH1)

# Filter for AIanxiety and True delta performance
filtered_coefficients_df <- combined_coefficients_df[combined_coefficients_df$term %in% c("AIAnxiety", "True_DeltaPerformance_Q0","AIAnxiety1", "True_DeltaPerformance_Q01"), ]


ggplot(filtered_coefficients_df, aes(x = term, y = estimate, color = model)) +
 geom_point(position = position_dodge(width = 1)) +  # Plot the point estimates, dodged for overlap
 geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.5, position = position_dodge(width = 1)) +  # Add error bars for CI
 labs(title = "Coefficients from the Tobit Models for AI Anxiety and True Delta Performance",
      x = "Predictors",
      y = "Coefficient Estimate") +
 coord_flip() +  # Flip the axes for better readability
 theme_minimal() +
 scale_color_manual(values = c("H1 True" = "darkblue", "No H1" = "orange"))  # Specify colors for each model
