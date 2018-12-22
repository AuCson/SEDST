# Explicit State Tracking with Semi-supervision for Neural Dialogue Generation

Code for CIKM'18 long paper: Explicit state tracking with semi-supervision for neural dialogue generation.

[Paper on Arxiv](https://arxiv.org/pdf/1808.10596.pdf)

## Requirements

The project was developed upon PyTorch 0.3.0 and now tested on PyTorch 0.4.0 and Python 3.6. 
- Experiments on task-orientend dialogues run well on a CPU.
- Experiments on non-task-oriented dialogues need GPUs to run.

## Dataset
### Task oriented datasets: 
- [Cambridge Restaurant Dataset](https://github.com/shawnwun/NNDIAL)
- [Stanford In-Car Personal Assistant Dataset](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/)

### Non-task oriented datasets:
- **Ubuntu Dialogue Corpus** Python pickled version can be downloaded [here](http://www.iulianserban.com/Files/UbuntuDialogueCorpus.zip) or at [julianser/Ubuntu-Multiresolution-Tools](https://github.com/julianser/Ubuntu-Multiresolution-Tools)
- **JD.com Customer Service Corpus** Please make a request as the description at [chenhongshen/HVMN](https://github.com/chenhongshen/HVMN)

## Running Experiments

For running the model, run:
```
python [semi_sup_model.py|unsup_model.py] -mode [train|adjust|test] -model [camrest|kvret|ubuntu|jd] -c spv_proportion=XXX OTHER_CONFIGS_IN_CONFIG_PY=VALUE
```
For running metrics, run:
```
python metric.py -f RESULT_FILE -t [camrest|kvret]
```
For running embedding base metrics, please follow [here](https://github.com/julianser/hed-dlg-truncated/tree/master/Evaluation)

## Directory

```
├───data
│   ├───CamRest676
│   ├───glove
│   ├───fasttext
│   ├───ubuntu
│   ├───jd
│   └───kvret
├───log
├───models
├───results
├───sheets
└───vocab
```

## Use Case
- **Task oriented dialogue systems**, such as restaurant reservation systems, where tracking user's intention and knowledge-base interactions are necessary, but you don't have enough(while still have some) labeled data for this.

- **Non-task oriented dialogue systems**, such as technique question answering conversations, where all the domain knowledge is embedded in the corpus and does not require knowledge base interaction. User's intention is completely not annotated, but the model can still extract context-aware "state spans" which helps response generation and visualization for humans. For example:

Turn |  User input | State span | Response |
--- | --- | --- | ---
0 | I just realised nothing can access my mysql database anymore , it says cannot connect , why would this be ? I did set it to be remote | remote would anymore mysql access | you can use the mysql client to connect to the server
1 | is it running ? | running I says mysql access | yes, I am running it now



## For Reference
Please consider citing:
```
@inproceedings{jin2018explicit,
  title={Explicit State Tracking with Semi-supervision for Neural Dialogue Generation},
  author={Jin, Xisen and Lei, Wenqiang and Ren, Zhaochun and Chen, Hongshen and Liang, Shangsong and Zhao, Yihong and Yin, Dawei},
  year={2018},
  booktitle={CIKM}
}

% additionally for experiments on task-oriented datasets

@inproceedings{lei2018sequicity,
  title={Sequicity: Simplifying task-oriented dialogue systems with single sequence-to-sequence architectures},
  author={Lei, Wenqiang and Jin, Xisen and Kan, Min-Yen and Ren, Zhaochun and He, Xiangnan and Yin, Dawei},
  booktitle={ACL},
  year={2018}
}
```

*Easter Egg: The last sentence of Section 6 in our ACL 2018 paper [Sequicity](https://www.comp.nus.edu.sg/~xiangnan/papers/acl18-sequicity.pdf)*

## Misc
- Some portion of the code is developed upon [Sequicity](https://github.com/WING-NUS/sequicity).

- Usually, you can get better results by fine-tuning the model with ``adjust`` option with a small learning rate after regular training ends. Since the size of two task-oriented datasets are relatively small, the final performance may fluctuate, depending on random seeds and hyperparameters. We reported the average of them. If you run two experiments with this code and default parameters with default fixed "0" random seed you will get the results below, where three of them are higher than the reported result in the paper.

 Metric/Dataset |Cambridge Restaurant 50% | In-Car Assistant 50% |
 -- | -- | --
 J.G. Acc | 95.14% | 81.89%
 Match |  93.58%  | 81.20%
