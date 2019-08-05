#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
    This code was modified by:
    @editor Patrick Brand
"""

import argparse
import os
import sys

import keras
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval import evaluate
from ..utils.keras_version import check_keras_version


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for evaluation.
    """
    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
        )
    elif args.dataset_type == 'pascal':
        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
        )
    elif args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, anchor_params=anchor_params)

    # print model summary
    # print(model.summary())

    # start evaluation
    if args.dataset_type == 'coco':
        from ..utils.coco_eval import evaluate_coco
        evaluate_coco(generator, model, args.score_threshold)
    else:
        average_precisions = evaluate(
            generator,
            model,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            save_path=args.save_path
        )

        # print evaluation
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        if sum(total_instances) == 0:
            print('No test instances found.')
            return

        print('mAP using the weighted average of precisions among classes: {:.4f}'.format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
        print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))
        
        return average_precisions


def evaluate_k_fold_experiment_models():   
    nr_kfold_train_test=5
    nr_inner_kfold_train_val=4
    data_base_folder = '/home/user/data/SPIE-retinanet/'
    experiment_path = '/mnt/synology/pelvis/projects/patrick/Experiments/SPIE_Anatomical_Prior/baseline'
    
    iou_thresholds = np.arange(0.5,1.0, 0.05)
    
    
    # Run experiments
    for fold_nr in range(nr_kfold_train_test):
        print('='*20)
        print(' RUNNING FOLD NUMBER: ', fold_nr)
        print('='*20)
        
        current_fold_path = os.path.join(experiment_path, 'fold_'+str(fold_nr))
        
        for inner_fold_nr in range(nr_inner_kfold_train_val):
            print(' '*4,'='*20)
            print(' '*4,' TRAIN/VAL INNER FOLD NUMBER: ', inner_fold_nr)
            print(' '*4,'='*20)
                
            # Create experiment directory for current inner train/val fold
            current_inner_fold_path = os.path.join(current_fold_path, 'inner_fold_'+str(inner_fold_nr))
            current_model_path = os.path.join(current_inner_fold_path, 'model')
                           
            CLASSES_PATH= os.path.join(data_base_folder, 'classes.csv')
            TEST_PATH= os.path.join(data_base_folder, 'folds/'+str(fold_nr)+'/test.csv')
            SNAPSHOT_PATH=current_model_path
            RESULTS_PATH=os.path.join(current_inner_fold_path, 'results')
            # Create directory for results
            if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)
            
            # Replace original paths to local copy paths
            old_path = '/mnt/synology/pelvis/projects/patrick/datasets/'
            new_path = '/home/user/data/'

            metadata = pd.read_csv(TEST_PATH, header=None)
            metadata = metadata.replace(regex=[old_path], value=new_path)
            metadata.to_csv(TEST_PATH, header=False, index=False)
            
            # Create results csv file
            csv_output_path = os.path.join(RESULTS_PATH, 'quantitative_results.csv')
            qualitative_output_path = os.path.join(RESULTS_PATH, 'qualitative_results')
            if not os.path.exists(qualitative_output_path):
                os.makedirs(os.path.join(RESULTS_PATH, 'qualitative_results'))
            
            results = pd.DataFrame(iou_thresholds, columns=['IoU_threshold'])
            
            # Gather results
            PZ_AP = []
            Prostate_AP = []
            mAP = []
            for current_threshold in iou_thresholds:  
                print(' '*8,'='*20)
                print(' '*8,'IoU threshold: {:.2f}'.format(current_threshold))
                print(' '*8,'='*20)
                arguments = ['csv',
                             TEST_PATH,
                             CLASSES_PATH,
                             os.path.join(current_model_path, 'resnet50_csv.h5'),
                             '--gpu=0', 
                             '--convert-model',
                             '--iou-threshold='+str(current_threshold),
                             '--max-detections=2']
                # Only generate visual results once
                if current_threshold == iou_thresholds[0]:
                    arguments.insert(4, '--save-path='+qualitative_output_path)
                
                # Calculate measurements for current IoU threshold
                average_precisions = main(args=arguments)
                
                current_pz_ap = average_precisions[0][0]
                current_prostate_ap = average_precisions[1][0]
                current_mAP = 0.5 * (current_pz_ap + current_prostate_ap)
                
                # Store current measurements
                PZ_AP.append(current_pz_ap)
                Prostate_AP.append(current_prostate_ap)
                mAP.append(current_mAP)
            
            # Store AP results in csv file
            results['PZ_AP'] = PZ_AP
            results['Prostate_AP'] = Prostate_AP
            results['mAP'] = mAP
            
            # Write results to disk
            results.to_csv(csv_output_path, index=False)



if __name__ == '__main__':
    fold_nr = sys.argv[1]
    inner_fold_nr = sys.argv[2]
    
    data_base_folder = '/home/user/data/SPIE-retinanet/'
    experiment_path = '/mnt/synology/pelvis/projects/patrick/Experiments/SPIE_Anatomical_Prior/baseline'
    
    iou_thresholds = np.arange(0.5,1.0, 0.05)
    
    # Run experiments
    print('='*20)
    print(' RUNNING FOLD NUMBER: ', fold_nr)
    print('='*20)

    current_fold_path = os.path.join(experiment_path, 'fold_'+str(fold_nr))


    print(' '*4,'='*20)
    print(' '*4,' TRAIN/VAL INNER FOLD NUMBER: ', inner_fold_nr)
    print(' '*4,'='*20)
        
    # Create experiment directory for current inner train/val fold
    current_inner_fold_path = os.path.join(current_fold_path, 'inner_fold_'+str(inner_fold_nr))
    current_model_path = os.path.join(current_inner_fold_path, 'model')
                    
    CLASSES_PATH= os.path.join(data_base_folder, 'classes.csv')
    TEST_PATH= os.path.join(data_base_folder, 'folds/'+str(fold_nr)+'/test.csv')
    SNAPSHOT_PATH=current_model_path
    RESULTS_PATH=os.path.join(current_inner_fold_path, 'results')
    # Create directory for results
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    
    # Replace original paths to local copy paths
    old_path = '/mnt/synology/pelvis/projects/patrick/datasets/'
    new_path = '/home/user/data/'

    metadata = pd.read_csv(TEST_PATH, header=None)
    metadata = metadata.replace(regex=[old_path], value=new_path)
    metadata.to_csv(TEST_PATH, header=False, index=False)
    
    # Create results csv file
    csv_output_path = os.path.join(RESULTS_PATH, 'quantitative_results.csv')
    qualitative_output_path = os.path.join(RESULTS_PATH, 'qualitative_results')
    if not os.path.exists(qualitative_output_path):
        os.makedirs(os.path.join(RESULTS_PATH, 'qualitative_results'))
    
    results = pd.DataFrame(iou_thresholds, columns=['IoU_threshold'])
    
    # Gather results
    PZ_AP = []
    Prostate_AP = []
    mAP = []
    for current_threshold in iou_thresholds:  
        print(' '*8,'='*20)
        print(' '*8,'IoU threshold: {:.2f}'.format(current_threshold))
        print(' '*8,'='*20)
        arguments = ['csv',
                        TEST_PATH,
                        CLASSES_PATH,
                        os.path.join(current_model_path, 'resnet50_csv.h5'),
                        '--gpu=0', 
                        '--convert-model',
                        '--iou-threshold='+str(current_threshold),
                        '--max-detections=2']
        # Only generate visual results once
        if current_threshold == iou_thresholds[0]:
            arguments.insert(4, '--save-path='+qualitative_output_path)
        
        # Calculate measurements for current IoU threshold
        average_precisions = main(args=arguments)
        
        current_pz_ap = average_precisions[0][0]
        current_prostate_ap = average_precisions[1][0]
        current_mAP = 0.5 * (current_pz_ap + current_prostate_ap)
        
        # Store current measurements
        PZ_AP.append(current_pz_ap)
        Prostate_AP.append(current_prostate_ap)
        mAP.append(current_mAP)
    
    # Store AP results in csv file
    results['PZ_AP'] = PZ_AP
    results['Prostate_AP'] = Prostate_AP
    results['mAP'] = mAP
    
    # Write results to disk
    results.to_csv(csv_output_path, index=False)
