import tensorflow as tf

model_path = r"D:\computer_vision_env\Computer_Vision_project\notebooks\cv_model_efficientNet.keras"
model = tf.keras.models.load_model(model_path)
print("Model summary:")
model.summary()
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
