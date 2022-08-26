import os
import tensorflow as tf

def translate(model, vn_food_name):
    input_text = tf.constant([vn_food_name])
    result = model.tf_translate(input_text)

    return result 

if __name__ == "__main__":
    model = tf.saved_model.load(os.path.join("models", "translator"))
    vn_food_name = "sữa tươi trân châu đường đen 4 tầng kem trứng nướng m"
    result = translate(model, vn_food_name)

    print(result['text'])