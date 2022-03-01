from bz2 import compress
from modulefinder import Module
import os, sys
import re
import logging
from simplejson import OrderedDict
from copy import deepcopy

import torch

from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch.initialization import register_default_init_args

import timm
from texttable import Texttable

import openvino

NOT_AVAILABLE_MESSAGE = 'N/A'

def get_model_list():
    full_list = timm.list_models()
    #model_list = [m for m in full_list if 'mobile' in m] # Shold be revised in the future
    #model_list += [m for m in full_list if 'vit' in m]
    #model_list = [m for m in full_list if 'levit' in m]
    #model_list = ['adv_inception_v3', 'bat_resnext26ts', 'beit_base_patch16_224','botnet26t_256', 'cait_m36_384', 'coat_lite_mini', 'convit_tiny', 'convmixer_768_32',  'convnext_base','crossvit_9_240', 'cspdarknet53', 'darknet53', 'deit_base_distilled_patch16_224','densenet121', 'dla34',  'dm_nfnet_f0',  'dpn68', 'eca_botnext26ts_256','ecaresnet26t', 'efficientnet_b0','efficientnet_el_pruned','efficientnet_lite0', 'efficientnetv2_l', 'ese_vovnet19b_dw','fbnetc_100', 'gcresnet33ts', 'gernet_l', 'gernet_m', 'gernet_s', 'ghostnet_050','gluon_senet154', 'gluon_seresnext50_32x4d', 'gluon_xception65', 'gmixer_12_224',  'gmlp_b16_224','halo2botnet50ts_256', 'hardcorenas_a','hrnet_w18', 'ig_resnext101_32x8d', 'inception_resnet_v2', 'inception_v3', 'inception_v4', 'jx_nest_base','lambda_resnet26rpt_256','lcnet_035','levit_128', 'mixer_b16_224','mnasnet_050', 'mobilenetv2_050', 'mobilenetv2_100','mobilenetv3_large_075', 'mobilenetv3_large_100','nasnetalarge', 'nest_base','nf_ecaresnet26', 'nf_ecaresnet50', 'nf_regnet_b0','nf_seresnet50', 'nfnet_f2s','pit_b_distilled_224', 'pit_s_224',  'pnasnet5large', 'regnetx_002','regnety_002',  'regnetz_b16','repvgg_a2', 'repvgg_b2',  'res2net50_14w_8s','resmlp_12_224','resmlp_36_224','resnest14d','resnet18', 'resnetblur18','resnetrs50','resnetv2_50d', 'resnetv2_50x1_bitm_in21k','resnext26ts','rexnetr_130','sebotnet33ts_256', 'sehalonet33ts', 'selecsls42','semnasnet_050',  'senet154', 'seresnet18', 'skresnet18', 'spnasnet_100', 'ssl_resnet18','swin_base_patch4_window7_224','swsl_resnet18','tresnet_m', 'tv_resnet34', 'twins_pcpvt_base','vgg11', 'visformer_small','vit_base_patch16_224','wide_resnet101_2', 'xception','xcit_large_24_p8_224']
    #model_list = ['mobilenetv2_050', 'repvgg_a2', 'vgg11', 'resnet18', 'resnest14d']
    #model_list = ['vgg11']
    model_list = ['gcresnet33ts']
    return model_list

def create_timm_model(name):
    model = timm.create_model(name, num_classes=1000, in_chans=3, pretrained=True, checkpoint_path='')
    return model

def export_to_onnx(model, save_here, model_name):
    x = torch.randn(1, 3, 224, 224, requires_grad=True)

    model_path = os.path.join(save_here, '{}.onnx'.format(model_name))

    torch.onnx.export(model,
                      x,
                      model_path,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=False)

    return model_path

def export_to_ir(model_path, save_here, model_name, prune=False):
    command_line = 'mo -m {}  -o {} -n {} --input_shape "(1,3,224,224)"'.format(model_path, save_here, model_name)
    if prune:
        command_line +=  " --transform Pruning"
    os.popen(command_line).read()

    return os.path.join(save_here, '{}.xml'.format(model_name)), os.path.join(save_here, '{}.bin'.format(model_name))
    

def optimize_with_nncf(model, save_here, model_name, config):
    # Quantize only for now
    onnx_model_path = os.path.join(save_here, '{}.onnx'.format(model_name))

    nncf_config = NNCFConfig.from_dict(config)
    compression_ctrl, model = create_compressed_model(model, nncf_config)
    compression_ctrl.export_model(onnx_model_path)

    return onnx_model_path

def run_benchmark(model_path):
    command_line = 'benchmark_app -m {} -d CPU '.format(model_path)
    output = os.popen(command_line).read()

    match = re.search("Throughput\: (.+?) FPS", output)
    if match != None:
        fps = match.group(1)
        return float(fps), output

    return None, output

def analyze_model(model_path):
    command_line = 'python model_analyzer/model_analyzer.py --model {} --ignore-unknown-layer'.format(model_path)
    output = os.popen(command_line).read()

    match1 = re.search("GFLOPs\: (.+?)\n", output)
    match2 = re.search("GIOPs\: (.+?)\n", output)
    if match1 != None and match2 != None:
        flops = float(match1.group(1))
        iops = float(match2.group(1))
        return iops/(flops+iops), output
    
    return None, output

def benchmark_model(model_path, dump_location, model_name, export_to_ov=False, prune=False, use_ma=False):
    """
    Receives the ONNX model and runs benchmarking tools with it
    """

    ops_ratio = NOT_AVAILABLE_MESSAGE
    weights_path = ""
    model_perf = NOT_AVAILABLE_MESSAGE

    try:
        if export_to_ov:
            model_path, weights_path = export_to_ir(model_path, dump_location, model_name, prune)
    
        if use_ma:
            ops_ratio, ouptut = analyze_model(model_path)
            if ops_ratio == None:
                logging.info("Failed to run Model Analyzer for model: {}\nDetails: {}\n".format(model_name, ouptut))
                ops_ratio = NOT_AVAILABLE_MESSAGE

        model_perf, bench_output = run_benchmark(model_path)

        if model_perf == None:
            logging.info("Cannot measure performance for the model: {}\nDetails: {}\n".format(model_name, bench_output))
            model_perf = NOT_AVAILABLE_MESSAGE
    except BaseException as error:
        logging.error("Error when becnhmarking the model: {}. Details: {}".format(model_name, error))
    else:
        if export_to_ov:
            cleanup([model_path, weights_path])

    return model_perf, ops_ratio
    
def cleanup(files):
    for file in files:
        os.remove(file)


nncf_configs = OrderedDict({
    "int8": {
        "input_info": {
        "sample_size": [1, 3, 224, 224]
        },
        "compression": [
            {
                "algorithm": "quantization",
                'quantize_inputs': True,
                'initializer': {
                    'range': {
                        'num_init_samples': 0
                    },
                    'batchnorm_adaptation': {
                        'num_bn_adaptation_samples': 0
                    }
                }
            }
        ]
    },
    "int8_pruning": {
        "input_info": {
        "sample_size": [1, 3, 224, 224]
        },
        "compression": [
            {
            "algorithm": "filter_pruning",
                "pruning_init": 0.4,
                "params": {
                "schedule": "exponential",
                    "pruning_target": 0.4,
                    "pruning_steps": 1,
                    "filter_importance": "geometric_median"
                }
            },
            {
                "algorithm": "quantization",
                'quantize_inputs': True,
                'initializer': {
                    'range': {
                        'num_init_samples': 0
                    },
                    'batchnorm_adaptation': {
                        'num_bn_adaptation_samples': 0
                    }
                }
            }
        ]
    }
})

def check_if_prune(nncf_config):
    prune = False
    for algo_config in nncf_config["compression"]:
        if "filter_pruning" in algo_config["algorithm"]:
            prune = True
            break
    return prune


def main():
    dump_location = sys.argv[1]

    logging.basicConfig(filename="log.txt",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

    model_list = get_model_list()

    logging.info("Optimizing models from the list: {}".format(model_list))

    table = Texttable()
    header = ["Model", "FP32 FPS"]
    for key in nncf_configs.keys():
        header.append("{} FPS".format(key))
        header.append("{} Speedup".format(key))
        header.append("{} Ops Ratio".format(key))

    table.header(header)
    
    for model_name in get_model_list():
        result = [NOT_AVAILABLE_MESSAGE] * len(table._header)
        result[0] = model_name

        model_files = []

        try:
            model = create_timm_model(model_name)

            orig_model_name = '{}_fp32'.format(model_name)
            orig_model_path = export_to_onnx(model, dump_location, orig_model_name)
            orig_model_perf, _ = benchmark_model(orig_model_path, dump_location, orig_model_name)
            result[1] = orig_model_perf
            model_files.append(orig_model_path)


            for i, (key, nncf_config) in enumerate(nncf_configs.items()):
                opt_model_name = '{}_{}'.format(model_name, key)
                opt_model_path = optimize_with_nncf(deepcopy(model), dump_location, opt_model_name, nncf_config)
                prune = check_if_prune(nncf_config)
                opt_model_perf, ops_ratio = benchmark_model(opt_model_path, dump_location, opt_model_name, True, prune, True)
                result[i * 3 + 2] = opt_model_perf
                result[i * 3 + 4] = ops_ratio

                speedup = opt_model_perf / orig_model_perf
                logging.info("Performance gain after applying optimizations to {}: {}".format(model_name, opt_model_perf / orig_model_perf))

                result[i * 3 + 3] = '{:.2f}x'.format(speedup)

                model_files.append(opt_model_path)

            
        except BaseException as error:
            logging.error("Unexpected error when optimizing model: {}. Details: {}".format(model_name, error))
        finally:
            cleanup(model_files) # Comment this to keep the resulted models

        table.add_row(result)

    logging.info(table.draw())
    print(table.draw())

if __name__ == "__main__":
    sys.exit(main() or 0)
