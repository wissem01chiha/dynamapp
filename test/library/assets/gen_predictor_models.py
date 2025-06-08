# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

import os
import shutil
import zipfile

import torch
from torch import nn

import tensorflow as tf
from tensorflow import keras


def _zip_and_delete_directory(directory_path):
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        print("Hello")
        raise ValueError(f"Directory does not exist: {directory_path}")

    # Construct the zip file name and its absolute path
    dirname = os.path.basename(directory_path)
    zip_filename = dirname + ".zip"
    abs_zip_path = os.path.join(os.getcwd(), zip_filename)

    try:
        # Create the zip file
        with zipfile.ZipFile(abs_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, directory_path)
                    zipf.write(file_path, arcname)
    finally:
        # Delete the original directory
        shutil.rmtree(directory_path)

    return abs_zip_path


def manage_models():
    # Create Torch model-1
    class Adder(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y

    model = Adder()
    model.eval()

    model_scripted = torch.jit.script(
        model,
        example_inputs=(
            1.0,
            2.0,
        ),
    )

    filename_torch_model_1 = "adder_1.pt"
    model_scripted.save(filename_torch_model_1)

    # Create Torch model-2
    class Adder2(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x + y, x

    model = Adder2()

    model.eval()

    model_scripted = torch.jit.script(
        model,
        example_inputs=(
            torch.ones(10, dtype=torch.float32),
            2.0 * torch.ones(10, dtype=torch.float32),
        ),
    )

    filename_torch_model_2 = "adder_2.pt"
    model_scripted.save(filename_torch_model_2)

    # Create Tensorflow model-1
    class AdderTF(keras.Model):
        def __init__(self):
            super(AdderTF, self).__init__()

        def call(self, inputs):
            x, y = inputs
            return x + y

    model = AdderTF()
    _ = model((tf.constant(1.0), tf.constant(2.0)))

    dirname_tf_model_1 = "adder_1"
    tf.saved_model.save(model, dirname_tf_model_1)
    abs_dir_path_tf_model_1 = os.path.abspath(dirname_tf_model_1)
    _ = _zip_and_delete_directory(abs_dir_path_tf_model_1)

    # Create Tensorflow model-2
    input_signature = [
        tf.TensorSpec(shape=(10), dtype=tf.float64),
        tf.TensorSpec(shape=(10), dtype=tf.float64),
    ]

    class Adder2_TF(keras.Model):
        def __init__(self):
            super(Adder2_TF, self).__init__()

        def call(self, x, y):
            return x + y, x

    model = Adder2_TF()

    # Calling model2 directly goes throgh keras preprocessing, which changes the input
    # types. To avoid this, create a tf.function and get a concrete function with
    # specified input shapes and types
    model_call_tf_func = tf.function(model.call)

    # Create a concrete function with specified input shapes and types
    input_signature = [
        tf.TensorSpec(shape=(None, 10), dtype=tf.float64),
        tf.TensorSpec(shape=(None, 10), dtype=tf.float64),
    ]
    concrete_function = model_call_tf_func.get_concrete_function(*input_signature)

    dirname_tf_model_2 = "adder_2"
    tf.saved_model.save(
        model, dirname_tf_model_2, signatures={"serving_default": concrete_function}
    )
    abs_dir_path_tf_model_2 = os.path.abspath(dirname_tf_model_2)
    _ = _zip_and_delete_directory(abs_dir_path_tf_model_2)


if __name__ == "__main__":
    manage_models()
