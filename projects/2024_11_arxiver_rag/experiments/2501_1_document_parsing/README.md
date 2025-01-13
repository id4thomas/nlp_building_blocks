# 2501_1_document_parsing
* 문서 내용 추출 & 레이아웃 분석 관련 내용 실험

## Experiments
### 1_arxiver_nougat
test md extraction code used in arxiver
* https://huggingface.co/datasets/neuralwork/arxiver
* https://github.com/neuralwork/arxiver
    * uses nougat-ocr model
    * https://github.com/facebookresearch/nougat

clone the nougat repostitory to circumvent following issue
* https://github.com/huggingface/transformers/issues/30029
* modify `nougat/model.py` - `BARTDecoder.prepare_inputs_for_inference`
    * `prepare_inputs_for_inference(..., **kwargs)`

### 2_mineru_test
* https://github.com/opendatalab/MinerU?tab=readme-ov-file

install requiremenss
* magic-pdf, pillow, paddlepaddle, doclayout-yolo, pycocotools, detectron2, ultralytics, unimernet, paddleocr, rapid-table, rapidocr_paddle, struct_eqtable
    * detectron2: `pip install 'git+https://github.com/facebookresearch/detectron2.git'`
* opencv

### 3_docling_test
* IBM docling
    * https://github.com/DS4SD/docling
    * https://github.com/DS4SD/docling-core

install requirements
* docling (uses torch)

### *_doclayout_yolov10
* use document layout yolo code
* https://github.com/opendatalab/DocLayout-YOLO?tab=readme-ov-file
    * https://huggingface.co/spaces/opendatalab/DocLayout-YOLO/blob/main/app.py

## Data Samples
* Attention Is All You Need
    * https://arxiv.org/abs/1706.03762v7


## Resources
OCR candidates
* https://github.com/OpenBMB/MiniCPM-V
* https://mindee.github.io/doctr/
* https://github.com/abgulati/kosmos-2_5-containerized?tab=readme-ov-file
* https://huggingface.co/NCSOFT/VARCO-VISION-14B
    * <ocr>
* https://github.com/VikParuchuri/surya?tab=readme-ov-file
* https://huggingface.co/cycloneboy/korean_PP-OCRv4_rec_infer
* https://huggingface.co/mPLUG/DocOwl2