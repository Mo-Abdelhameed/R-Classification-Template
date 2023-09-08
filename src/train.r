# Required Libraries
library(jsonlite)
library(dplyr)
library(tidyr)
library(caret)
library(readr)
library(data.table)
library(fastDummies)
library(nnet)

# Define directories and paths

ROOT_DIR <- getwd()
MODEL_INPUTS_OUTPUTS <- file.path(ROOT_DIR, 'model_inputs_outputs')
INPUT_DIR <- file.path(MODEL_INPUTS_OUTPUTS, "inputs")
INPUT_SCHEMA_DIR <- file.path(INPUT_DIR, "schema")
DATA_DIR <- file.path(INPUT_DIR, "data")
TRAIN_DIR <- file.path(DATA_DIR, "training")
MODEL_ARTIFACTS_PATH <- file.path(MODEL_INPUTS_OUTPUTS, "model", "artifacts")
OHE_ENCODER_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'ohe.RDS')
PREDICTOR_FILE_PATH <- file.path(MODEL_ARTIFACTS_PATH, "predictor", "predictor.RDS")
IMPUTATION_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'imputation.RDS')
LABEL_ENCODER_FILE <- file.path(MODEL_ARTIFACTS_PATH, 'label_encoder.RDS')
ENCODED_TARGET_FILE <- file.path(MODEL_ARTIFACTS_PATH, "encoded_target.RDS")


if (!dir.exists(MODEL_ARTIFACTS_PATH)) {
    dir.create(MODEL_ARTIFACTS_PATH, recursive = TRUE)
}
if (!dir.exists(file.path(MODEL_ARTIFACTS_PATH, "predictor"))) {
    dir.create(file.path(MODEL_ARTIFACTS_PATH, "predictor"))
}

# Reading the schema
file_name <- list.files(INPUT_SCHEMA_DIR, pattern = "*.json")[1]
schema <- fromJSON(file.path(INPUT_SCHEMA_DIR, file_name))
features <- schema$features

numeric_features <- features$name[features$dataType == "NUMERIC"]
categorical_features <- features$name[features$dataType == "CATEGORICAL"]
id_feature <- schema$id$name
target_feature <- schema$target$name
model_category <- schema$modelCategory


# Reading training data
file_name <- list.files(TRAIN_DIR, pattern = "*.csv")[1]
df <- read.csv(file.path(TRAIN_DIR, file_name), na.strings = c("", "NA", "?"))

# Impute missing data
imputation_values <- list()

columns_with_missing_values <- colnames(df)[apply(df, 2, anyNA)]
for (column in columns_with_missing_values) {
    if (column %in% numeric_features) {
        value <- median(df[, column], na.rm = TRUE)
    } else {
        value <- as.character(df[, column] %>% tidyr::replace_na())
        value <- value[1]
    }
    df[, column][is.na(df[, column])] <- value
    imputation_values[column] <- value
}
saveRDS(imputation_values, IMPUTATION_FILE)


# Encoding Categorical features
ids <- df[, id_feature]
target <- df[, target_feature]
df <- df %>% select(-all_of(c(id_feature, target_feature)))

# One Hot Encoding
if(length(categorical_features) > 0){
    df_encoded <- dummy_cols(df, select_columns = categorical_features, remove_selected_columns = TRUE)
    encoded_columns <- setdiff(colnames(df_encoded), colnames(df))
    saveRDS(encoded_columns, OHE_ENCODER_FILE)
    df <- df_encoded
}

# Label encoding target feature
levels_target <- levels(factor(target))
encoded_target <- as.integer(factor(target, levels = levels_target)) - 1
saveRDS(levels_target, LABEL_ENCODER_FILE)
saveRDS(encoded_target, ENCODED_TARGET_FILE)

# Train the Classifier

if (model_category == 'binary_classification'){
    model <- glm(target ~ ., family = binomial(link = "logit"), data = df)

} else if (model_category == "multiclass_classification") {
   model <- multinom(target ~ ., data = df)
}
saveRDS(model, PREDICTOR_FILE_PATH)
