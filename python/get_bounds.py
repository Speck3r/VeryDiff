
from logging import warning
import os 
import sys 

from collections import OrderedDict

import numpy as np
import h5py

import onnx
from onnx2pytorch import ConvertModel
import onnx_graphsurgeon as gs

import torch
from torch.onnx import _constants

_constants.ONNX_DEFAULT_OPSET = 20  # want to support gelu
torch.onnx.utils.GLOBALS.export_onnx_opset_version = 20

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))         
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../alpha-beta-CROWN/auto_LiRPA')))

torch.set_default_dtype(torch.float64)

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA import register_custom_op

from gelu_relaxation import BoundGeluTight

import argparse
import warnings


def load_onnx_model(onnx_path, bound_opts=None, dtype=torch.float32):
    onnx_model = onnx.load(onnx_path)

    shape_dict = extract_input_shapes(onnx_model)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch_model = ConvertModel(onnx_model)

    if dtype == torch.float64:
        torch_model = torch_model.double()
    elif dtype == torch.float32:
        torch_model = torch_model.float()

    x_concrete = tuple([torch.zeros(sh, dtype=dtype) for sh in shape_dict.values()])

    with warnings.catch_warnings():
        # abCROWN also uses onnx2pytorch, which triggers TracerWarning
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        model = BoundedModule(torch_model, x_concrete, bound_opts=bound_opts)

    return model.to(dtype)


def redirect_out(onnx_path, output_path, output_name):
    onnx_model = onnx.load(onnx_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    graph = gs.import_onnx(onnx_model)
    
    # specify output 
    tensors = [t for n in graph.nodes for t in n.outputs if t.name == output_name]
    graph.outputs = [tensors[0]]
    graph.cleanup().toposort()
    onnx_model_eps_out = gs.export_onnx(graph)

    onnx.save(onnx_model_eps_out, output_path)


def extract_input_shapes(onnx_model):
    shape_dict = OrderedDict()
    for node in onnx_model.graph.input:
        name = node.name 
        shape = []

        for dim in node.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)  # e.g. 'batch_size'
            else:
                shape.append(None)  # unkown shape

        shape_dict[name] = shape 

    return shape_dict


def compute_pre_activation_bounds(onnx_path, bounds_path, output_name, outfile, method='alpha-crown', tight_gelu=True, dtype=np.float64):
    if tight_gelu:
        register_custom_op("onnx::Gelu", BoundGeluTight)

    # should already be the model with the error inputs
    onnx_model = onnx.load(onnx_path)

    shape_dict = extract_input_shapes(onnx_model)

    path_redirect = f"redirect_out_{os.path.basename(onnx_path)}"
    redirect_out(onnx_path, path_redirect, output_name)

    def read_bound_widths(file, key):
        if key in file.keys():
            return file[key]
        else:
            return np.zeros(shape_dict[key])

    with h5py.File(bounds_path, "r") as f:
        input_vars = []
        for k, v in shape_dict.items():
            arr = read_bound_widths(f, k)[:].astype(dtype)
            # for input, we have 2 as initial shape!
            # so just assume that everything has the correct shape!
            # arr = arr.reshape(v)

            if arr.shape[0] == 2:
                # we only have upper and lower bounds for the input
                data_lb = torch.tensor(arr[0]).unsqueeze(0)
                data_ub = torch.tensor(arr[1]).unsqueeze(0)
            else:
                data_lb = -torch.tensor(arr)
                data_ub =  torch.tensor(arr)

            ptb = PerturbationLpNorm(x_L=data_lb, x_U=data_ub)
            center = 0.5 * (data_lb + data_ub)
            bounded_x = BoundedTensor(center, ptb)

            input_vars.append(bounded_x)

    model = load_onnx_model(path_redirect, dtype=torch.float64 if dtype == np.float64 else torch.float32)

    lb, ub = model.compute_bounds(x=tuple(input_vars), method=method)

    #print("lb: ", lb)
    #print("ub: ", ub)

    bounds = torch.concat((lb, ub)).detach().numpy()

    with h5py.File(outfile, "w") as f:
        f.create_dataset(output_name, data=bounds)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Use auto_LiRPA to compute bounds for neural networks where actiation functions introduce a separate error term.")
    parser.add_argument('onnx_path', type=str, help="Path to onnx model that should be analyzed (already containing error inputs!)")
    parser.add_argument('input_file', type=str, help="Path to hdf5 file containing bounds for relevant inputs (unlisted inputs are assumed to be 0)")
    parser.add_argument('output_name', type=str, help="Name of onnx output to compute bounds for")
    parser.add_argument('--output_file', type=str, default="out_bounds.hdf5", help="File to store lower and upper bounds of output (default: out_bounds.hdf5)")
    parser.add_argument('--tight_gelu', action='store_true', help="Use improved initialization of GeLU relaxation.")
    args = parser.parse_args()

    compute_pre_activation_bounds(args.onnx_path, args.input_file, args.output_name, args.output_file, method='alpha-crown', 
                                  tight_gelu=args.tight_gelu, dtype=np.float64)

    



