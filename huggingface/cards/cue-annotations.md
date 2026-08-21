---
license: apache-2.0
language:
- en
task_categories:
- text-generation
size_categories:
- 100K<n<1M
tags:
- cue
- user-simulation
- persona
- annotations
configs:
- config_name: ABCD
  data_files:
  - split: train
    path: ABCD/train-*
  - split: validation
    path: ABCD/validation-*
- config_name: AirDialogue
  data_files:
  - split: train
    path: AirDialogue/train-*
  - split: validation
    path: AirDialogue/validation-*
- config_name: BiTOD
  data_files:
  - split: train
    path: BiTOD/train-*
  - split: validation
    path: BiTOD/validation-*
- config_name: CaSiNo
  data_files:
  - split: train
    path: CaSiNo/train-*
  - split: validation
    path: CaSiNo/validation-*
- config_name: CraigslistBargains
  data_files:
  - split: train
    path: CraigslistBargains/train-*
  - split: validation
    path: CraigslistBargains/validation-*
- config_name: DSTC2-Clean
  data_files:
  - split: train
    path: DSTC2-Clean/train-*
  - split: validation
    path: DSTC2-Clean/validation-*
- config_name: Disambiguation
  data_files:
  - split: train
    path: Disambiguation/train-*
  - split: validation
    path: Disambiguation/validation-*
- config_name: FRAMES
  data_files:
  - split: train
    path: FRAMES/train-*
- config_name: GECOR
  data_files:
  - split: train
    path: GECOR/train-*
- config_name: HDSA-Dialog
  data_files:
  - split: train
    path: HDSA-Dialog/train-*
  - split: validation
    path: HDSA-Dialog/validation-*
- config_name: KETOD
  data_files:
  - split: train
    path: KETOD/train-*
  - split: validation
    path: KETOD/validation-*
- config_name: KVRET
  data_files:
  - split: train
    path: KVRET/train-*
  - split: validation
    path: KVRET/validation-*
- config_name: MS-DC
  data_files:
  - split: train
    path: MS-DC/train-*
- config_name: MULTIWOZ2_2
  data_files:
  - split: train
    path: MULTIWOZ2_2/train-*
  - split: validation
    path: MULTIWOZ2_2/validation-*
- config_name: MetaLWOZ
  data_files:
  - split: train
    path: MetaLWOZ/train-*
- config_name: MuDoCo
  data_files:
  - split: train
    path: MuDoCo/train-*
  - split: validation
    path: MuDoCo/validation-*
- config_name: MulDoGO
  data_files:
  - split: train
    path: MulDoGO/train-*
  - split: validation
    path: MulDoGO/validation-*
- config_name: MultiWOZ_2.1
  data_files:
  - split: train
    path: MultiWOZ_2.1/train-*
  - split: validation
    path: MultiWOZ_2.1/validation-*
- config_name: SGD
  data_files:
  - split: train
    path: SGD/train-*
  - split: validation
    path: SGD/validation-*
- config_name: STAR
  data_files:
  - split: train
    path: STAR/train-*
- config_name: Taskmaster1
  data_files:
  - split: train
    path: Taskmaster1/train-*
  - split: validation
    path: Taskmaster1/validation-*
- config_name: Taskmaster2
  data_files:
  - split: train
    path: Taskmaster2/train-*
- config_name: Taskmaster3
  data_files:
  - split: train
    path: Taskmaster3/train-*
- config_name: WOZ2_0
  data_files:
  - split: train
    path: WOZ2_0/train-*
- config_name: all
  data_files:
  - split: train
    path: all/train-*
  - split: validation
    path: all/validation-*
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
- config_name: lmsys-chat-1m
  data_files:
  - split: train
    path: lmsys-chat-1m/train-*
- config_name: wildchat
  data_files:
  - split: train
    path: wildchat/train-*
dataset_info:
- config_name: ABCD
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 38742938
    num_examples: 7710
  - name: validation
    num_bytes: 4681318
    num_examples: 948
  download_size: 12369635
  dataset_size: 43424256
- config_name: AirDialogue
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 26924010
    num_examples: 5212
  - name: validation
    num_bytes: 26465654
    num_examples: 5187
  download_size: 14156179
  dataset_size: 53389664
- config_name: BiTOD
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 15063587
    num_examples: 2839
  - name: validation
    num_bytes: 1498110
    num_examples: 286
  download_size: 4677499
  dataset_size: 16561697
- config_name: CaSiNo
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 4517906
    num_examples: 820
  - name: validation
    num_bytes: 150925
    num_examples: 28
  download_size: 1413806
  dataset_size: 4668831
- config_name: CraigslistBargains
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 21117887
    num_examples: 3898
  - name: validation
    num_bytes: 2969731
    num_examples: 556
  download_size: 6602104
  dataset_size: 24087618
- config_name: DSTC2-Clean
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 8479899
    num_examples: 1612
  - name: validation
    num_bytes: 2626022
    num_examples: 506
  download_size: 2562239
  dataset_size: 11105921
- config_name: Disambiguation
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 47128487
    num_examples: 8375
  - name: validation
    num_bytes: 5643168
    num_examples: 995
  download_size: 14032977
  dataset_size: 52771655
- config_name: FRAMES
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 6586952
    num_examples: 1248
  download_size: 1946643
  dataset_size: 6586952
- config_name: GECOR
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 3288491
    num_examples: 676
  download_size: 855586
  dataset_size: 3288491
- config_name: HDSA-Dialog
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 46921429
    num_examples: 8395
  - name: validation
    num_bytes: 5626913
    num_examples: 995
  download_size: 14155953
  dataset_size: 52548342
- config_name: KETOD
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 21503336
    num_examples: 4029
  - name: validation
    num_bytes: 2643656
    num_examples: 504
  download_size: 6759110
  dataset_size: 24146992
- config_name: KVRET
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 10790461
    num_examples: 2355
  - name: validation
    num_bytes: 1322638
    num_examples: 295
  download_size: 3186185
  dataset_size: 12113099
- config_name: MS-DC
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 47764378
    num_examples: 9917
  download_size: 12661675
  dataset_size: 47764378
- config_name: MULTIWOZ2_2
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 45900483
    num_examples: 8391
  - name: validation
    num_bytes: 5497075
    num_examples: 996
  download_size: 14237269
  dataset_size: 51397558
- config_name: MetaLWOZ
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 49982648
    num_examples: 9783
  download_size: 13267199
  dataset_size: 49982648
- config_name: MuDoCo
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 29905020
    num_examples: 6013
  - name: validation
    num_bytes: 3358824
    num_examples: 685
  download_size: 8407062
  dataset_size: 33263844
- config_name: MulDoGO
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 49324693
    num_examples: 9901
  - name: validation
    num_bytes: 5566455
    num_examples: 1133
  download_size: 13895749
  dataset_size: 54891148
- config_name: MultiWOZ_2.1
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 46219416
    num_examples: 8382
  - name: validation
    num_bytes: 5535166
    num_examples: 994
  download_size: 14230413
  dataset_size: 51754582
- config_name: SGD
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 49699095
    num_examples: 9493
  - name: validation
    num_bytes: 11762552
    num_examples: 2276
  download_size: 16997475
  dataset_size: 61461647
- config_name: STAR
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 28617484
    num_examples: 5590
  download_size: 8327047
  dataset_size: 28617484
- config_name: Taskmaster1
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 34022640
    num_examples: 6045
  - name: validation
    num_bytes: 4162268
    num_examples: 752
  download_size: 10395597
  dataset_size: 38184908
- config_name: Taskmaster2
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 55961314
    num_examples: 9813
  download_size: 14522955
  dataset_size: 55961314
- config_name: Taskmaster3
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 54503937
    num_examples: 9728
  download_size: 14680547
  dataset_size: 54503937
- config_name: WOZ2_0
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 2963677
    num_examples: 600
  download_size: 790298
  dataset_size: 2963677
- config_name: all
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 848664187
    num_examples: 160531
  - name: validation
    num_bytes: 89510475
    num_examples: 17136
  download_size: 255758248
  dataset_size: 938174662
- config_name: default
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: string
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 848664187
    num_examples: 160531
  - name: validation
    num_bytes: 89510475
    num_examples: 17136
  download_size: 255758094
  dataset_size: 938174662
- config_name: lmsys-chat-1m
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: 'null'
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 49076679
    num_examples: 9909
  download_size: 14713818
  dataset_size: 49076679
- config_name: wildchat
  features:
  - name: dataset
    dtype: string
  - name: split
    dtype: string
  - name: source_repo
    dtype: string
  - name: source_config
    dtype: 'null'
  - name: source_split
    dtype: string
  - name: source_revision
    dtype: 'null'
  - name: source_id
    dtype: string
  - name: source_native_id
    dtype: string
  - name: turns_sha256
    dtype: string
  - name: persona_manual
    dtype: string
  - name: annotation_version
    dtype: int64
  - name: provenance
    dtype: string
  - name: metadata
    dtype: string
  splits:
  - name: train
    num_bytes: 53558449
    num_examples: 9797
  download_size: 16159079
  dataset_size: 53558449
---

# CUE annotations

533,001 persona-manual annotations over 28 public dialogue corpora, one config per corpus, split
`train` / `validation`. Each row describes how the user in one conversation behaves.

## No dialogue text is redistributed

Rows carry provenance and a hash, **not** the source turns:

| column | meaning |
|--------|---------|
| `persona_manual` | the annotation, a JSON string (`json.loads` it) |
| `source_repo`, `source_config`, `source_split`, `source_id`, `source_native_id` | where the conversation came from |
| `turns_sha256` | hash of the turns the annotation was derived from |
| `annotation_version`, `provenance`, `metadata` | JSON strings with the annotator settings |

To train on this you fetch each conversation from its source dataset and join on `source_id`,
then check `turns_sha256` to confirm you reconstructed the same turns. The Apache-2.0 license
covers these annotations only — **every source corpus keeps its own license and terms**, listed
per row in `source_repo`.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("handshake-ai-research/cue-annotations", "ABCD", split="train")
manual = json.loads(ds[0]["persona_manual"])
```
