import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('retinal-oct.h5')
model_upgrade = tf.keras.models.load_model(
    'retinal-oct_10epochs_32batchsize_GPUTraining32Workers_Opti_Ftrl.h5')

test_data_dir = 'content/OCT2017/test/'
img_width, img_height = 150, 150
batch_size = 32

valid_test_datagen = ImageDataGenerator(
    rescale=1./255,
)

test_generator = valid_test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
)


(eval_loss, eval_accuracy) = model.evaluate(
    test_generator, batch_size=batch_size)
print('Test Loss: ', eval_loss)
print('Test Accuracy: ', eval_accuracy)

(eval_loss_upgrade, eval_accuracy_upgrade) = model_upgrade.evaluate(
    test_generator, batch_size=batch_size)
print('Test Loss: ', eval_loss_upgrade)
print('Test Accuracy: ', eval_accuracy_upgrade)


def non_regression_test_loss(eval_loss, eval_loss_upgrade):
    assert (eval_loss >= eval_loss_upgrade) == True
    return ("Loss amélioré!")


def non_regression_test_accuracy(eval_accuracy, eval_accuracy_upgrade):
    assert (eval_accuracy >= eval_accuracy_upgrade) == True
    return ("Accuracy amélioré!")


non_regression_test_loss(eval_loss, eval_loss_upgrade)
non_regression_test_accuracy(eval_accuracy, eval_accuracy_upgrade)
