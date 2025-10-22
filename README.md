Official implementation of “A Unified Multi-Task Framework for Out-of-domain Plant Disease Classification and Severity Grading via Unlabeled Domain Adaptation”.
---

### Environment
------

You can create a new Conda environment by running the following command:

```bash
conda env create -f environment.yml
```

In the environment, we still use other networks. If it does not work, please configure the environment of other networks first.

### PreTrained Model
------

The pre-trained Uniplant-CG model is linked below, you can download it.

-  [Uniplant-CG]()
- And we use ViT, the download is [here]().

### Train
------

```
bash train.sh
```

### Test
------

```
bash test.sh
```

