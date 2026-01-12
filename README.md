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

-  Uniplant-CG: [Here](http://118.195.249.181:8888/download?filename=%2Fwww%2Fwwwroot%2Fdata%2FUniplantCG%2Funi.zip) are the trained Uniplant-CG weights.
- And we use ViT, the download is [here](http://118.195.249.181:8888/download?filename=%2Fwww%2Fwwwroot%2Fdata%2FUniplantCG%2Fjx_vit_base_p16_224-80ecf9dd.pth).

### Data
- The data will be available upon request after the paper is published.

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

