---
title: "Weather Classification"
author: "Theo Hilsen"
---

## Fragestellung & Datenmenge

## Libraries


rm(list = ls())


## Libraries


#Laden der benoetigten Pakete 
library(magick)
library(keras)
library(deepANN)
library(marray)


## Import der Datenmenge


#Datenvorverarbeitung
#Bilder sind auf 150x150 zugeschnitten. Aus der dataset2 Datei
#werden die Pfade genommen und die Bilder als Liste gespeichert, welche fürs
#trainieren vorgesehen sind.

# Name des Datensatzes: Multi-class Weather Dataset for Image Classification
# Downloadlink https://data.mendeley.com/datasets/4drtyfjtfy/1

#Basisdirectory
data_dir <- "/Users/thi/Desktop/KI/dataset2"

# Liste aller Dateien im Ordner image_dir, die mit ".jpg" enden
filelist <- list.files(data_dir, pattern = "\\.jpg$", full.names = TRUE)



## Transformation der Datenmenge


#Bilddimensionen
height   <- 150L
width    <- 150L 
channels <- 3L



# Transformieren der Datenmenge mit magick

# Transformation in einen 2D Array mit den vorgegbenen Dimensionen und in Farbe
magick_array <- function(img, channels) {
  as.integer(magick::image_data(img, channels))
}

# Anpassung der Unterschiednlichen Bildgroessen
magick_resize <- function(img, height, width) {
  magick::image_scale(img, magick::geometry_size_pixels(width = width, height = height, preserve_aspect = FALSE))
}

# Verwandelt die vorverarbeiteten daten in einen 4D Tensor lesbaren Array (Fuer X)
images <- images_load(filelist, FUN = magick::image_read) %>% 
  images_resize(FUN = magick_resize, h = height, w = width) %>%
  as_images_array(FUN = magick_array, ch = "rgb") %>% 
  as_images_tensor(height = height, width = width, channels = channels)


# Extrahiere die Dateinamen aus den Dateinamen ohne Endung(Fuer Y)
classnames_raw <- sapply(strsplit(gsub("\\.jpg$", "", filelist), "/"), function(x) x[length(x) ])
#Extrahiere daraus die Klassennamen
classnames <- gsub("\\d+$", "", classnames_raw)

#Factorisierung
labels <- factor(classnames)

#One-Hot Format:
Y <- as_CNN_image_Y(labels)


## Visialisierung & Laden der Datenmenge


# Vorschau der ersten 10 Bilder mit iherer Klasifikation von der Datenmenge.
for (i in 1:10) {
  nr <- i * 10
  plot(as.raster(images[nr, , , ],max = 256L))
  classnames_raw <- classnames_raw[as.numeric(labels[nr])]
  title(paste("Image", i, "Class:", classnames_raw))
}



## CNN Model definition

## Unser CNN-Model besteht aus einer ersten convolutions Layer mit einem Input-Shape von height = 150, width = 150 und channels = 3, um die Bilder fuer das Model in der Richtigen Interpretationsform bereitzustellen. Wir haben hier erst einmal mit einem filter von 64 angefangen und einer Kernel_Size von 3x3, da die verarbeitung so von groß nach klein geht wie in der darauf folgenden Layer mit 32 Filtern dann zu sehen ist. Beim Modelieren haben wir dann nach einigen Iterationen und veraenderten Hyperparametern noch jeweils eine convolutions Layer mit 64 Filtern und 32 Filtern hinzugefuegt. So konnten wir unsere Accuracy um einiges verbessern, da sich das Model durch eine hoeher Batch_size und die angepassten Filter in einem relativ kleinen Datensatz besser trainieren konnte. Unsere Max-Pooling-Layer haben wir nach jeder convolutions Layer eingebaut um die dimensionalität der Ausgabe der vorherigen Convolutional-Schicht reduziert. Die Max-Pooling-Layer haben wir auf eine pool_size von 2x2 gesetz um wiederum den kleinen Datensatz und die somit limitierten trainigs des Models zu verbessern. Zum Ende des Models wird noch eine Flatten_Layer zur transformation des outputs der Convolutions-Layer genutzt. Die Dens-Layer beginnen mit 128 neuronen um eine erste klassifikation vorzunehmen. Diese werden dann auf vier Moeglichkeiten reduziert. Zwischen den Dense layers wir noch einmal die Haelfte der neuronen rausgenommen um so weiteres Overfitting zu vermeiden


# Model definition
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", input_shape = c(height, width, channels)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = nunits(Y), activation = "softmax")

#Uberblick des Models 
summary(model)


## Model Kompilierung & Training


# Kompilierung
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Fitting des Models fuer die Train-Daten mit 0.20 an Test-Daten
history <- model %>% fit(
  x = images,
  y = Y,
  batch_size = 32,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2
)


## Performance des Models


# Visualisierung des Models
plot(history)



# Evaulation nach Train loss & Accuracy
scores <- model %>% evaluate(
  images, Y, verbose = 1
)
cat('Train loss:', scores[[1]], '\n')
cat('Train accuracy:', scores[[2]], '\n')


## Wir haben eine In-Sample-Prediciton genommen, damit keine weiteren Bilddaten erhoben werden müssen


#In-Sample-predict

yhat_prob <- model %>% predict(images)

#Klassennamen extrahieren
classname <- colnames(Y)
#Vorhergesagte Klassen aneinanderreihen
yhat_label <- classname[apply(yhat_prob, 1L, which.max)]
#Die vorhergesagten Wahrscheinlichkeiten aneinanderreihen
yhat_label_idx <- apply(yhat_prob, 1L, which.max)

#Wahrscheinlihckeit dafür, dass etwas richtig vorhergesagt wurde
deepANN::accuracy(labels, yhat_label)

#Spalte bauen
prob_per_label <- sapply(seq_len(NROW(yhat_prob)), function(i) {
  yhat_prob[i, yhat_label_idx[i]]
})
#Matrix erstellen
prob_matrix <- cbind(Probability = prob_per_label, Predicted_class = yhat_label, Actual = as.character(labels))



# Uebersciht der In-Sampel-prediction
head(prob_matrix)


## Model zum Benchmarking (Naivebayes Model)


library(naivebayes)
#Selber SPlit wie bei #cnn-Modell bei 80%, damit beide vergleichbar bleiben
tr_split <- sample(1:length(labels), round(length(labels) * 0.8))
#Labels und images bei diesem split trennen
tr_images <- images[tr_split, , , ]
tr_labels <- labels[tr_split]
test_images <- images[-tr_split, , , ]
test_labels <- labels[-tr_split]


# Trainieren des Naive-Bayes-Modells
nb_model <- naivebayes::naive_bayes(tr_images, tr_labels)

summary(nb_model)

nb_prediction <- predict(nb_model, matrix(test_images))

#Accuracy berechnen für das Naive-Bayes Modell
accuracy <- mean(nb_prediction == test_labels)
print(paste0("Accuracy: ", round(accuracy * 100, 2), "%"))



## Vergleich:


#Vergleich ziehen:
print(paste0("Accuracy Naive-Bayes: ", round(accuracy * 100, 2), "%"))
cat('Accuracy CNN', scores[[2]], '\n')
#Das neuronale CNN-Modell ist deutlich genauer als das Naive-Bayes Modell. Das liegt daran, dass beim Naive-Bayes Modell alle Merkmale als unabhängig voneinander gelten, während beim CNN Muster erkannt und diese kombiniert werden. Deswegen werden Muster nicht erkannt von dem Naive-Bayes Modell.

