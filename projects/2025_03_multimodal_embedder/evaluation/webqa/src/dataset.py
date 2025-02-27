import base64
import json
from typing import List

from io import BytesIO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pydantic import BaseModel

from torch.utils.data import Dataset

class WebQAImageData(BaseModel):
    image_id: int
    title: str
    caption: str
    url: str
    imgUrl: str

class WebQAData(BaseModel):
    Q: str
    A: List[str]
    split: str
    Guid: str
    # images: List[WebQAImageData]
    image: WebQAImageData

def load_webqa_dataset(fpath) -> List[WebQAData]:
    '''WebQA_test.json'''
    dataset = json.load(open(fpath, "r"))

    datas = []
    for sample in dataset.values():
        img_facts = [
            WebQAImageData(**x) for x in sample["img_Facts"]
        ]
        ## Flatten
        for img_fact in img_facts:
            data = WebQAData(
                Q=sample["Q"],
                A=sample["A"],
                split=sample["split"],
                Guid=sample["Guid"],
                # images=img_facts
                image=img_fact
            )
            datas.append(data)
    return datas

class WebQADataset(Dataset):
    def __init__(
        self,
        data_fpath: str,
        lineidx_fpath: str,
        images_fpath: str,
        seed: int = 42
    ):
        self.datas = load_webqa_dataset(data_fpath)
        with open(lineidx_fpath, "r") as fp_lineidx:
            self.lineidxs = [int(i.strip()) for i in fp_lineidx.readlines()]

        self.images_fpath = images_fpath

    def load_image(self, image_data: WebQAImageData) -> Image.Image:
        img_id = image_data.image_id
        with open(self.images_fpath, "r") as fp:
            fp.seek(self.lineidxs[int(img_id)%10000000])
            imgid, img_base64 = fp.readline().strip().split('\t')
        im = Image.open(BytesIO(base64.b64decode(img_base64)))    
        return im
    
    def __getitem__(self, idx):
        sample = self.datas[idx]
        image = self.load_image(sample.image)

        candidate_text = "\n".join(
            [sample.image.title, sample.image.caption]
        )
        return {
            "query": sample.Q,
            "candidate_text": candidate_text,
            "candidate_image": image
        }

    def __len__(self):
        return len(self.datas)