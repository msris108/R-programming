require(keras)
############

fashion_mnist <- dataset_fashion_mnist()

#Test Train Split
#train_images <- fashion_mnist$train$x
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Explore data structure
dim(train_images)
str(train_images)

#Plotting the image
fobject <- train_images[9,,]
plot(as.raster(fobject, max = 255))

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

class_names[train_labels[9]+1]


#Normalizing [(X-mean)/Std.Dev]

train_images <- train_images / 255
test_images <- test_images / 255


#Creating a validation split - used for hyperparameter tuning
val_indices <- 1:5000
val_images <- train_images[val_indices,,]
part_train_images <- train_images[-val_indices,,]
val_labels <- train_labels[val_indices]
part_train_labels <- train_labels[-val_indices]


# Flattening
# X X X
# Y Y Y  -> X X X Y Y Y Z Z Z
# Z Z Z

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'sgd', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)
# Sparse_categorical_crossentropy => more than 2 classes and observation can belong to only one class
# Binary_crossentropy => 2 classes and object belongs to one of the two classes
# Categorical_crossentropy => more than 2 classes and observation can belong to multiple classes


model %>% fit(part_train_images, part_train_labels, epochs = 30, batch_size=100, validation_data=list(val_images,val_labels))



# Test Performance

score <- model %>% evaluate(test_images, test_labels)

cat('Test loss:', score$loss, "\n")
cat('Test accuracy:', score$acc, "\n")

# Predicting on Test set

predictions <- model %>% predict(test_images)
predictions[1, ]
which.max(predictions[1, ])
class_names[which.max(predictions[1, ])]
plot(as.raster(test_images[1, , ], max = 1))

class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]