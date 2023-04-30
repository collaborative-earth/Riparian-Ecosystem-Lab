import tensorflow
import onnx
from onnx_tf.backend import prepare

if __name__ == '__main__':
    path_to_onnx = "/Users/cmurray/projects/collaborative-earth/pretrained-models/SatMAE/fmow_vit_large_patch16.onnx"
    path_to_tf = "/Users/cmurray/projects/collaborative-earth/pretrained-models/SatMAE/fmow_vit_large_patch16.tf"
    path_to_keras = "/Users/cmurray/projects/collaborative-earth/pretrained-models/SatMAE/fmow_vit_large_patch16.keras"

    model_onnx = onnx.load(path_to_onnx)
    tf_rep = prepare(model_onnx)
    tensorflow_model_rep = tf_rep.export_graph(path_to_tf)

    model = tensorflow.keras.models.load_model(path_to_tf)
    # model.save(path_to_keras)