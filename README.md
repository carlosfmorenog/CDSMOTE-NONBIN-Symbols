# CDSMOTE-NONBIN-Symbols

An imbalance handling method to improve the detection and classification of symbols depicted in Piping and Instrumentation Diagrams (P&amp;IDs)

See `demo.ipynb` for a jupyter notebook demo on how to use this method.

The original symbol dataset can be downloaded from here:
[https://www.dropbox.com/s/sj277k4slmrv3qc/symbols_combined_pixel_red.csv?dl=0](https://www.dropbox.com/s/sj277k4slmrv3qc/symbols_combined_pixel_red.csv?dl=0)

The generated symbol dataset, once CDSMOTE has been applied, can be downloaded from here:
[https://www.dropbox.com/s/ll562q3gjqyrhp9/cdsmotedb_symbols_combined_pixel_kmeans.csv?dl=0](https://www.dropbox.com/s/ll562q3gjqyrhp9/cdsmotedb_symbols_combined_pixel_kmeans.csv?dl=0)

Please reference this method as follows:

* E. Elyan, C. F. Moreno-García & P. Johnston, “Symbols in Engineering Drawings (SiED): An imbalanced dataset benchmarked by convolutional neural networks”, Engineering Applications of Neural Networks (EANN) 2020, Halkidiki, Greece, INNS 2, pp. 215–224. [https://doi.org/10.1007/978-3-030-48791-1_16](https://doi.org/10.1007/978-3-030-48791-1_16).

* L. Jamieson, C. F. Moreno-García & E. Elyan, “A multiclass imbalanced dataset classification of symbols from piping and instrumentation diagrams”, In: Barney Smith, E.H., Liwicki, M., Peng, L. (eds). International Conference on Document Analysis and Recognition (ICDAR 2024). Lecture Notes in Computer Science, vol 14804, pp. 3-16. Springer, Cham. [https://doi.org/10.1007/978-3-031-70533-5_1](https://doi.org/10.1007/978-3-031-70533-5_1).

or use the BibTex entries below:

@inproceedings{Elyan2020,
author = {Elyan, Eyad and Moreno-Garc{\'{i}}a, Carlos Francisco and Johnston, Pamela},
booktitle = {Engineering Applications of Neural Networks (EANN)},
doi = {10.1007/978-3-030-48791-1},
isbn = {9783030487911},
keywords = {Imbalance,P&ID,classification,cnn,engineering drawings,id,imbalanced dataset,multiclass,p},
mendeley-tags = {Imbalance,P&ID},
pages = {215--224},
title = {{Symbols in Engineering Drawings (SiED): An Imbalanced Dataset Benchmarked by Convolutional Neural Networks}},
year = {2020}
}

@inproceedings{Jamieson2024,
author = {Jamieson, Laura and Moreno-Garc{\'{i}}a, Carlos Francisco and Elyan, Eyad},
booktitle = {International Conference on Document Analysis and Recognition (ICDAR)},
doi = {10.1007/978-3-031-70533-5},
isbn = {9783031705335},
keywords = {convolutional neural networks,piping and instrumentation diagrams},
pages = {3--16},
title = {{A Multiclass Imbalanced Dataset Classification of Symbols from Piping and Instrumentation Diagrams}},
year = {2024}
}
