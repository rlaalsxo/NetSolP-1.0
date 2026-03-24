# Copyright (c) 2021, Technical University of Denmark
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from data import *
import os
import pickle
import argparse
import pandas as pd
import time
import numpy as np


def sigmoid(x): return 1 / (1 + np.exp(-x))

MAX_SEQ_LEN = 1024


def _fix_onnx_dynamic_shapes(model_path):
    static_path = model_path.replace(".onnx", "_static.onnx")
    import onnx
    from onnx import shape_inference
    model = onnx.load(model_path)
    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param == "batch_size":
                dim.dim_value = 1
                dim.dim_param = ""
            elif dim.dim_param == "seq_len":
                dim.dim_value = MAX_SEQ_LEN
                dim.dim_param = ""
    for out in model.graph.output:
        for dim in out.type.tensor_type.shape.dim:
            if dim.dim_param == "batch_size":
                dim.dim_value = 1
                dim.dim_param = ""
    model = shape_inference.infer_shapes(model)
    onnx.save(model, static_path)
    return static_path


def create_sessions(model_paths, args):
    if args.PROVIDER == "furiosa":
        from furiosa.runtime.sync import create_runner
        static_paths = [_fix_onnx_dynamic_shapes(mp) for mp in model_paths]
        return [create_runner(sp) for sp in static_paths]
    import onnxruntime
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = args.NUM_THREADS
    opts.inter_op_num_threads = args.NUM_THREADS
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if torch.cuda.is_available() else ["CPUExecutionProvider"])
    return [onnxruntime.InferenceSession(mp, sess_options=opts, providers=providers)
            for mp in model_paths]


def _pad_to_static(toks, np_mask):
    seq_len = toks.shape[1]
    if seq_len >= MAX_SEQ_LEN:
        return toks[:, :MAX_SEQ_LEN], np_mask[:, :MAX_SEQ_LEN]
    pad_len = MAX_SEQ_LEN - seq_len
    toks_padded = torch.nn.functional.pad(toks, (0, pad_len), value=1)
    mask_padded = torch.nn.functional.pad(np_mask, (0, pad_len), value=False)
    return toks_padded, mask_padded


def run_inference(sessions, toks, lengths, np_mask, args):
    if args.PROVIDER == "furiosa":
        toks_p, mask_p = _pad_to_static(toks, np_mask)
        inputs = [toks_p.numpy(), lengths.numpy(), mask_p.numpy()]
        return [s.run(inputs)[0] for s in sessions]
    inputs_names = sessions[0].get_inputs()
    ort_inputs = {
        inputs_names[0].name: toks.numpy(),
        inputs_names[1].name: lengths.numpy(),
        inputs_names[2].name: np_mask.numpy()
    }
    return [s.run(None, ort_inputs)[0] for s in sessions]


def close_sessions(sessions, args):
    if args.PROVIDER == "furiosa":
        for s in sessions:
            s.close()


def get_preds_split(split_i, embed_dataloader, args, prediction_type, test_df):
    if args.MODEL_TYPE == "Both":
        model_types = ["ESM12", "ESM1b"]
    else:
        model_types = [args.MODEL_TYPE]

    model_paths = [os.path.join(args.MODELS_PATH, f"{prediction_type}_{mt}_{split_i}_quantized.onnx") for mt in model_types]
    sessions = create_sessions(model_paths, args)

    embed_dict = {}
    for i, (toks, lengths, np_mask, labels) in enumerate(embed_dataloader):
          outs = run_inference(sessions, toks, lengths, np_mask, args)
          embed_dict[labels[0]] = sum(outs) / len(outs)

    close_sessions(sessions, args)

    pred_df = pd.DataFrame(embed_dict.items(), columns=["sid", "preds"])
    pred_df = test_df.merge(pred_df)
    return pred_df


def run_model_distilled(embed_dataloader, args, prediction_type, test_df):
    model_paths = [os.path.join(args.MODELS_PATH,
      f"{prediction_type}_ESM1b_distilled_quantized.onnx")]
    sessions = create_sessions(model_paths, args)

    embed_dict = {}
    for i, (toks, lengths, np_mask, labels) in enumerate(embed_dataloader):
          outs = run_inference(sessions, toks, lengths, np_mask, args)
          embed_dict[labels[0]] = sum(outs) / len(outs)

    close_sessions(sessions, args)

    pred_df = pd.DataFrame(embed_dict.items(), columns=['sid', 'preds'])
    pred_df = test_df.merge(pred_df)
    return pred_df


def get_preds(args):
    fasta_dict = read_fasta(args.FASTA_PATH)
    test_df = pd.DataFrame(fasta_dict.items(), columns=['sid', 'fasta'])
    test_df["fasta"] = test_df["fasta"].apply(lambda x: x[:1022])
    print(len(test_df))

    alphabet_path = os.path.join(args.MODELS_PATH,
      f"ESM12_alphabet.pkl")

    with open(alphabet_path, "rb") as f:
        alphabet = pickle.load(f)
    embed_dataset = FastaBatchedDataset(test_df)
    embed_batches = embed_dataset.get_batch_indices(0, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(alphabet), batch_sampler=embed_batches)

    if "S" in args.PREDICTION_TYPE:
        print("Doing Solubility")
        preds_per_split = []
        for i in range(5):
            print(f"Model {i}")
            pred_df = get_preds_split(i, embed_dataloader, args, "Solubility", test_df)
            preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
            preds_per_split.append(preds_i)
            test_df[f"predicted_solubility_model_{i}"] = preds_i
        avg_pred = sum(preds_per_split) / 5
        test_df["predicted_solubility"] = pd.Series(avg_pred)
    if "U" in args.PREDICTION_TYPE:
        print("Doing Usability")
        preds_per_split = []
        for i in range(5):
            print(f"Model {i}")
            pred_df = get_preds_split(i, embed_dataloader, args, "Usability", test_df)
            preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
            preds_per_split.append(preds_i)
            test_df[f"predicted_usability_model_{i}"] = preds_i
        avg_pred = sum(preds_per_split) / 5
        test_df["predicted_usability"] = pd.Series(avg_pred)

    test_df.to_csv(args.OUTPUT_PATH, index=False)


def get_preds_distilled(args):
    fasta_dict = read_fasta(args.FASTA_PATH)
    test_df = pd.DataFrame(fasta_dict.items(), columns=['sid', 'fasta'])
    test_df["fasta"] = test_df["fasta"].apply(lambda x: x[:1022])
    print(len(test_df))

    alphabet_path = os.path.join(args.MODELS_PATH,
      f"ESM12_alphabet.pkl")

    with open(alphabet_path, "rb") as f:
        alphabet = pickle.load(f)
    embed_dataset = FastaBatchedDataset(test_df)
    embed_batches = embed_dataset.get_batch_indices(0, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(alphabet), batch_sampler=embed_batches)

    if "S" in args.PREDICTION_TYPE:
        print("Doing Solubility")
        pred_df = run_model_distilled(embed_dataloader, args, "Solubility", test_df)
        preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
        test_df[f"predicted_solubility"] = preds_i
    if "U" in args.PREDICTION_TYPE:
        print("Doing Usability")
        pred_df = run_model_distilled(embed_dataloader, args, "Usability", test_df)
        preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
        test_df[f"predicted_usability"] = preds_i

    test_df.to_csv(args.OUTPUT_PATH, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--FASTA_PATH", type=str, help="Input protein sequences in the FASTA format"
    )
    parser.add_argument(
        "--OUTPUT_PATH", type=str, help="Output location of predictions as CSV"
    )
    parser.add_argument(
        "--MODELS_PATH", default="./models/", type=str, help="Location of models"
    )
    parser.add_argument(
        "--NUM_THREADS",
        default=os.cpu_count(),
        type=int,
        help="Number of threads to use. (Use more for faster results)"
    )
    parser.add_argument(
        "--MODEL_TYPE",
        default="ESM1b",
        choices=['ESM12', 'ESM1b', 'Both', 'Distilled'],
        type=str,
        help="Model to use. ESM1b is better but much slower. Both option averages the prediction"
    )
    parser.add_argument(
        "--PREDICTION_TYPE",
        default="S",
        choices=['S', 'U', 'SU'],
        type=str,
        help="Either Solubility(S), Usability(U) or Both"
    )
    parser.add_argument(
        "--PROVIDER",
        default="onnx",
        choices=['onnx', 'furiosa'],
        type=str,
        help="Inference backend. 'onnx' for ONNX Runtime (CPU/CUDA), 'furiosa' for FuriosaAI RNGD NPU"
    )
    args = parser.parse_args()
    t1 = time.time()
    if args.MODEL_TYPE == "Distilled":
        get_preds_distilled(args)
    else:
        get_preds(args)
    print(f"Finished prediction in {time.time()-t1}s")
