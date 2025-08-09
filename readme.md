# CE_CTC model baseline for JBM555

This is the source code of the baseline CE_CTC model for the following paper:

Yu Sugimoto, Jun-You Wang, Li Su, and Eita Nakamura, "Singing MIDI Transcription with Music Language Models: Formulation and Comparison," submitted to APSIPA 2025.

Basically, it is a fork of my own [TASLP paper's source code back in 2023](https://github.com/york135/CTC_CE_for_AST), where we proposed to use both the cross-entropy loss and the CTC loss to train a note-level singing voice transcription (i.e., singing MIDI transcription) model. I did some modifications so that the model can be trained with the JBM555 dataset as discussed in the APSIPA paper.

**Main differences from my TASLP work in 2023:**

- Use Spleeter for singing voice separation -> use **HT Demucs (Demucs v4)** for singing voice separation.

- Use the MIR-ST500 dataset for training and evaluation -> use the proposed **JBM555 dataset** for training, validation, and evaluation.

- Search for the best onset/silence threshold and model checkpoint on the training set -> search for the best onset/silence threshold and model checkpoint on the **validation split of JBM555**

- Some path configurations are modified.

These settings make the CE+CTC model a fairly competitive singing MIDI transcription model even in 2025. It serves as a baseline for our APSIPA work.

It is possible to run all the source code on python 3.7 and 3.8. I didn't test the code on python 3.9 or above, but I think those environments should also be OK.

## Install dependencies

The simplest way to install all dependencies is:

```
pip install -r requirements.txt
```

If this does not work due to some install order issues, then try this command:

```
cat requirements.txt | xargs -n 1 pip install
```

The above command will also install *mir_eval*, *matplotlib*, *madmom*, and other packages that are only required for model evaluation, plotting results, or other parts that are not related to inference. If you do not want to install those packages, just modify `requirements.txt` to remove these dependencies.

## Dataset preparation

I assume the dataset is organized as follows:

```
[dataset_dir]
|-- a3JRV466Ci.wav
|-- A41zTifbyq.wav
`-- ...
```

### Run source separation

```
python run_demucs_v4.py [dataset_dir] [separated_dir]
```

The output vocal wav files will be written to `[separated_dir]`, with the same data format as `[dataset_dir]`. 

### Convert the dataset to the MIR-ST500 format

```
python build_dataset_format.py [dataset_dir] [separated_dir] \
 [output_dir]
```

where `[output_dir]` here is a directory to store the dataset with a similar file structure format as MIR-ST500.

### Feature extraction

Please first modify `gen_feature.yaml` by setting `audio_dir` to your `[output_dir]` at the last step. Then:

```
cd feature_extraction
python generate_feature.py gen_feature.yaml
```

### Separate train/validation/test split

```
python split_dataset.py [dataset_dir]
```

The `dataset_dir` here should be the same as the `output_dir` in the last step. Then, it will create three folders under `dataset_dir`: `JBM555_train`, `JBM555_validation`, and `JBM555_test`, and then put the songs into the folders of their respective splits.

## Create groundtruth data

### For strongly-labeled data

```
python gt_strong_dataset.py [gt_json_path] [output_pickle_path] \
    [dataset_dir] [feature_file_name] [weighting_method]
```

This should be executed three times (for training/validation/test set). Each time, the `dataset_dir` should be set to `[dataset_dir]/JBM555_train`, `[dataset_dir]/JBM555_validation`, and `[dataset_dir]/JBM555_test`, where `[dataset_dir]` here is the `dataset_dir` in the last step.

Besides, `gt_json_path` should be `JBM555_trim_note_annotation.json`; `output_pickle_path` specifies the path to store the generated groundtruth pickle file; `feature_file_name` should be the same as the `feature_output_name` in the `gen_feature.yaml` config file. `weighting_method` should be `CE_CTC`.

### For weakly-labeled data

```
python gt_weak_dataset.py [gt_json_path] [output_pickle_path] \
 [dataset_dir]
```

Similarly, this should be executed three times (for training/validation/test set).

## Model training

Then, modify  Line 4 to 11 of `config/train_jbm.yaml` to proper paths and directories, and run the following command:

```
python train.py [yaml_path] [device] [--pretrained_path]
```

where `yaml_path` should be `config/train_jbm.yaml`, `device` is the device used for training (cpu, cuda:0, etc), `--pretrained_path` is an optional argument that specifies the path to the pretrained model.

## Hyper-parameter search

After defining these arguments, run the following command to obtain raw predictions for each epoch's checkpoints. Before this, edit the `dataset_dir` of `search_param_config_jbm.yaml` to the directory of the validation set.

```
cd search_param
python predict_each_epoch.py [yaml_path] [device] \
    [prediction_output_prefix] [model_path_prefix]
```

`predict_each_epoch.py` loads every model checkpoints and inference the validation set. Instead of generating the transcription results, it dumps the models' frame-level prediction (onset probability, silence probability, etc) to pickle files.

Here, `yaml_path` is the path to the config file which should be set to `search_param_config_jbm.yaml`; `device` is the device used for inference, `prediction_output_prefix` specifies the prefix of the path to dump the model predictions, `model_path_prefix` specifies the prefix of model checkpoints (e.g., `../models/jbm555`).

Then, we run the following command to try different sets of parameters for post-processing, which will then determine the best set of parameters:

```
python find_parameter.py [yaml_path] [prediction_output_prefix] \
    [json_best_threshold_output_path] [model_performance_output_path]
```

Again, `yaml_path` is the path to the config file which should be set to `search_param_config_jbm.yaml`; `prediction_output_prefix` specifies the prefix of the pickle files where the model predictions are at. After hyper-parameter searching, the best onset and offset threshold for each model checkpoint will be written to `json_best_threshold_output_path`, and the model performance using these best thresholds will be written to `model_performance_output_path` in pickle format.

Then, read the best set of hyperparameter with:

```
python read_training_set_result.py [model_performance_output_path] \
 [json_best_threshold_output_path]
```

It will show something like:

```
Best epoch 93 onset threshold = 0.4 offset threshold = 0.7000000000000001
         Precision Recall F1-score
COnPOff  0.592963 0.595312 0.591867
COnP     0.770172 0.771418 0.767279
COn      0.859859 0.859422 0.855440
gt note num: 46868.0 tr note num: 46359.0
```

These are the performance metrics on the validation set. Then, modify the `onset_thres` and `offset_thres` of `config/inference_jbm.yaml` to this set of hyperparameters.

## Evaluation on the test set

Finally, to test the model on the **test set**, run:

```
python predict.py [yaml_path] [dataset_dir] [h5py_feature_file_name] \
    [predict_json_file] [model_path] [device]
```

where `yaml_path` is the path to the config yaml file, which should be set to `config/inference_jbm.yaml`; `model_path` is the path to the best model checkpoint, `device` specifies the device used for singing transcription.

Again, `dataset_dir` is the directory to the test dataset, `h5py_feature_file_name` specifies the name of h5 feature files, `predict_json_file` is the desired path where the output JSON file will be written to.

Finally, to obtain the COn, COnP, and COnPOff F1-scores, run the following command:

```
python evaluate.py [gt_file] [predicted_file] [tol]
```

where `gt_file` is the path to the groundtruth note labels (JSON file, which should be `JBM555_trim_note_annotation.json` in this case), `predicted_file` is the path to the predicted JSON file, `tol` is the onset tolerance, which is usually set to `0.05` (i.e., 50ms).

This python program will then compute the model performance.

## Inference one song

```
python do_everything.py [input_path] [output_path] \
    [model_path] [yaml_path] [device]
```

where `input_path` is the path to the audio file to be transcribed; `output_path` is the path to the MIDI file that the transcription will be written to; `model_path` is the path to the model checkpoint for singing transcription; `yaml_path` is the path to the config yaml file; `device` specifies the device used to perform singing transcription. To use our APSIPA baseline version (pretrained model), `model_path` should be set to `models/jbm555_80`, and `yaml_path` should be set to `config/inference_jbm.yaml`.

For example:

```
python do_everything.py mysong.mp4 mysong.mid \
  models/jbm555_80 configs/inference_jbm.yaml cuda:0
```
