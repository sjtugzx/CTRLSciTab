# CTRLSciTab
Code and data for Paper "TOWARDS CONTROLLED TABLE-TO-TEXT GENERATION WITH SCIENTIFIC REASONING".

## Dataset
CTRLSciTab, a specialized dataset in the scientific domain, comprises 8,967 pairs of tables and their corresponding descriptions. This dataset introduces a novel task: controlled table-to-text generation underpinned by scientific reasoning. The central objective of this task is to produce analytical narratives that not only adhere to the specific knowledge within a given domain but also align with predefined user preferences. It is anticipated that CTRLSciTab will provide a valuable benchmark for future research endeavors in the area of controlled table-to-text generation that incorporates scientific reasoning.

## Getting Started

### Requirements

```
pip install -r requirements.txt
```

## Dataset Description
The CTRLSciTab dataset consists of three .json files, where each line is a JSON dictionary with the following format:
```
{
  "annotated_highlight_cell": [
    "BRAM (Kbits)"
  ],
  "annotated_highlight_pos": [
    [
      1,
      0
    ]
  ],
  "descriptions": "Due to lack of big FC layers and lower total number of parameters, ResNet requires fewer BRAMs than AlexNet.",
  "domain_specific_knowledge": "As for resource utilization, as shown in Table 3,ResNet-18 requires 75% more LUTs, which is thereason we were forced to divide it into three DFEs. For example,AlexNet requires 724M MACs for the processing of asingle image . In addition, since the DFE platform allows usto easily split the network into multiple FPGAs, we canimplement even larger networks, such as ResNet andAlexNet. For example, increasing the size of input from32 32 to 96 96 increases the resource utilization byapproximately 5% for all types of resources.7.7.1 Our theoretical estimation of the number of clocks perpicture for ResNet-18 (the largest network implemented)is approximately 1:85 106. Additionally, we implemented the AlexNet , sinceit is one of the most well-known DNNs and is often usedas a basis for new techniques in DNNs such as networkcompression, performance improvements, and new typesof layers       . For our evaluation, we implementedResNet-18, AlexNet and a VGG-like CNN, based onone proposed by Umuroglu et al. The rst big success of NNs inImageNet classi cations is AlexNet . CNN models used in our evaluations (ResNet andAlexNet) are based on the work of Hubara et al. a Our implementation of ResNet-18 consumes 5 less power and is 4 slower for ImageNet, when compared to the same NN on the latest Nvidia GPUs. AlexNet is aDNN consisting of ve convolutional, three pooling andthree fully connected (FC) layers, which uses ReLU foractivation.2.3 Inferencing and especially training NNs requires a lotof multiply-accumulate operations (MACs) to computethe weighted sums of the neurons' inputs. All this allowsus to run a full-sized ResNet-18 and AlexNet on twoand three FPGAs, respectively, achieving runtimecomparable with the latest GPUs, consuming less powerand energy. In our paper, we present an implementationof a full-sized AlexNet for the ImageNet dataset (224224 3) using only on-chip memory. Using the Winogradtransform , they achieved state-of-the-art results onthe AlexNet architecture. For AlexNet (input size224 224), the power consumption of the DFE increases,since three DFEs are needed to t the network. , which achieved top-1 accuracyof up to 51:2% for ImageNet, using ResNet-18-basedBNN. Both AlexNet and ResNet can handle r 1000-class real-time classi cation on an FPGA. In our implementation, max pooling is used inall cases, except for the last pooling in ResNet-18.5.2.3 As was shown in FINN , BatchNorm and one-bitactivation can be replaced by a threshold function. For the full-sizedImageNet dataset of size 224 224, we used theResNet18 and AlexNet model, while for other inputs we used aVGG-like CNN, based on one used by Umuroglu et al.,as its topology is more suitable for the above-mentioneddatasets. Dueto lack of big FC layers and lower total number ofparameters, ResNet requires fewer BRAMs than AlexNet.6.2.3 We compared our implementation with FINN byUmuroglu et al. Most successful CNN models such asAlexNet , GoogLeNet , ResNet  and Inception can be used to classify thousands of di erent objectswith high accuracy. In contrast to previous works, we use 2-bit ac:v tivations instead of 1-bit ones, which improves AlexNet's i top-1 accuracy from 41:8% to 51:03% for the ImageNet X classi cation. Skip connections forward the output of onelayer to the one after adjacent one, skipping one layer.This resolves the vanishing gradient  problem, thusincreasing the number of layers and achievingstate-ofthe-art accuracy on image-related problems   .We developed a hardware design for skip connectionsand, to analyze their performance, implemented theResNet-18  network, which architecture is shown inTable 1. On a DFE,ResNet-18 takes only 17:5% more time for inference,while for GPUs this number is 42:5%. While this is not helpful inrealtime applications, it can speed up the process if a largeamount of already-available data must be processed.6.2.2 To analyze the e ect of adding skip connections andincreasing network depth, we compared the performanceof AlexNet and ResNet on DFE. 7 In addition, we implemented a full-sized quantized 1 AlexNet.",
  "paper_id": "1708.00052",
  "table_caption": "TABLE III: Comparison of ResNet-18 and AlexNet networks",
  "table_column_names": [
    "[EMPTY]",
    "AlexNet",
    "ResNet-18"
  ],
  "table_content_values": [
    [
      "LUT",
      "343295",
      "596081"
    ],
    [
      "BRAM (Kbits)",
      "34600",
      "30854"
    ],
    [
      "FF",
      "664767",
      "1175373"
    ],
    [
      "Run time (ms)",
      "13.7",
      "16.1"
    ]
  ],
  "table_highlight_cell": [
    "BRAM Kbits",
    "FF"
  ],
  "table_highlight_cell_pos": [
    [
      1,
      0
    ],
    [
      2,
      0
    ]
  ],
  "title": "Streaming Architecture for Large-Scale Quantized Neural Networks on an FPGA-Based Dataflow Platform"
}
```

-The table metadata consists of the title, <mark>table_caption</mark>, and strings to provide the model with more context about the table.

-The content of the tabular data includes <mark>table_column_names</mark>, <mark>table_content_values</mark>, and descriptions.

-The <mark>table_highlight_cell</mark> field is a list of strings that denote the automatically detected cells. The <mark>table_highlight_cell_pos</mark> is a list of [[row_index, column_index]] pairs, each pair indicating that 'table[row_index][column_index]' denotes the position of a <mark>table_highlight_cell</mark>.

-The <mark>annotated_highlight_cell</mark> field is a list of strings that denote the cells annotated by experts. The <mark>annotated_highlight_pos</mark> is a list of [[row_index, column_index]] pairs, where each pair indicates that <mark>table[row_index][column_index]</mark> denotes the position of an <mark>annotated_highlight_cell</mark>.

-The <mark>paper_id</mark> is a unique identifier for this example.

-The <mark>domain_specific_knowledge</mark> indicates domain-specific knowledge extracted from the original PDF file.


## Official Task
This task aims to generate natural language descriptions that are both fluent and accurate, incorporating domain-specific knowledge while remaining consistent with the tabular data and user preferences. The input consists of structured data, highlighted cells, and domain-specific knowledge, denoted as $D={(T, H, B)}$. Here, $T$ signifies a linearized table, with $T = \left\{t_1,\cdots, t_{|n|}\right\}$. Each tabular data, $t_i$, consists of an attribute-value pair, where $a_i$ and $v_i$ can take values such as strings, numbers, phrases, or sentences. The highlighted cells are akin to $t_i$ and are denoted by $H = \left\{h_1,\cdots, h_{|n|}\right\}$, acting as prompts reflecting user preferences. Furthermore, $B = \left\{b_i,\cdots,b_{|m|}\right\}$ represents domain-specific knowledge, with each $b_i$ corresponding to a sentence associated with the tabular data. The expected output is an analytical description aligned with user preferences and incorporating domain-specific knowledge $R$.

## Train / Dev / Test Scripts
All relevant code is available in the 'Baselines' folder.





