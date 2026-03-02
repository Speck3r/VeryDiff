
import onnx 
import onnx_graphsurgeon as gs

import argparse

def insert_error_nodes(onnx_path, output_path, activation_layers=None):
    if activation_layers is None:
        activation_layers = ['Relu', 'Gelu']

    onnx_model = onnx.load(onnx_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    graph = gs.import_onnx(onnx_model)

    for node in graph.nodes:
        if node.op in activation_layers:
            assert len(node.outputs) == 1, f"We assume 1d output for activations! Got {len(node.outputs)}"
            old_out = node.outputs[0]
            
            # create new input for the error
            eps_input = gs.Variable(name=f"eps_{node.name}", dtype=old_out.dtype, shape=old_out.shape)
            graph.inputs.append(eps_input)
    
            # add eps to old output of activation function
            new_out = gs.Variable(name=f"{old_out.name}_pre_error", dtype=old_out.dtype, shape=old_out.shape)
            node.outputs = [new_out]
    
            add_node = gs.Node(op="Add", inputs=[new_out, eps_input], outputs=[old_out])
            graph.nodes.append(add_node)
    
    graph.cleanup().toposort()
    onnx_model_eps = gs.export_onnx(graph)

    onnx.save(onnx_model_eps, output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add error inputs after each activation layer to an onnx model.")
    parser.add_argument('onnx_path', type=str, help="onnx model to add error inputs to")
    parser.add_argument('output_path', type=str, help="path to store modified onnx model to")
    args = parser.parse_args()

    insert_error_nodes(args.onnx_path, args.output_path)