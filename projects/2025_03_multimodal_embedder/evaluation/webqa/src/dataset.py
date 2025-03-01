import base64
import copy
import json
from typing import List, Literal, Optional, Tuple

from io import BytesIO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pydantic import BaseModel, Field

from torch.utils.data import Dataset

# class WebQAImageData(BaseModel):
#     image_id: int
#     title: str
#     caption: str
#     url: str
#     imgUrl: str

# class WebQAData(BaseModel):
#     Q: str
#     A: List[str]
#     split: str
#     Guid: str
#     # images: List[WebQAImageData]
#     image: WebQAImageData

# class ImageData(BaseModel):
#     fpath: str

class WebQAQuery(BaseModel):
    id: str
    query: str
    positive_ids: List[str] = Field([], description="positive candidate ids")
    
class WebQACandidate(BaseModel):
    id: str
    modality: Literal["t", "ti"] = "t"
    text: Optional[str] = None

def load_webqa_data(
    fpath,
    task: Literal["t2t", "t2ti"]="t2t",
    text_template = "{title} {fact}",
    image_text_template = "{title} {caption}"
) -> Tuple[List[WebQAQuery], List[WebQACandidate]]:
    '''WebQA_test.json'''
    dataset = json.load(open(fpath, "r"))

    queries = []
    candidates = []
    for query_id, data in dataset.items():
        query = WebQAQuery(
            id=str(query_id),
            query=data["Q"]
        )
        query_candidates = []
        if task=="t2ti":
            for candidate_data in data["img_Facts"]:
                candidate = WebQACandidate(
                    id=str(candidate_data["image_id"]),
                    modality="ti",
                    text=image_text_template.format(
                        title=candidate_data["title"],
                        caption=candidate_data["caption"],
                    )
                )
                query_candidates.append(candidate)
        elif task=="t2t":
            for candidate_data in data["txt_Facts"]:
                candidate = WebQACandidate(
                    id=str(candidate_data["snippet_id"]),
                    modality="t",
                    text=text_template.format(
                        title=candidate_data["title"],
                        fact=candidate_data["fact"],
                    )
                )
                query_candidates.append(candidate)
        else:
            raise ValueError(f"task {task} not supported")
        query.positive_ids = [x.id for x in query_candidates]
        
        queries.append(query)
        candidates.extend(query_candidates)
    return queries, candidates

class WebQAQueryDataset(Dataset):
    def __init__(
        self,
        data: List[WebQAQuery]
    ):
        self.data=data
        
    def __getitem__(self, idx):
        data = self.data[idx]
        return {"text": data.query}
    
    def __len__(self):
        return len(self.data)

class WebQATCandidateDataset(Dataset):
    """Text only candidates"""
    def __init__(
        self,
        data: List[WebQACandidate],
    ):
        self.data=data

    def __getitem__(self, idx):
        data = self.data[idx]
        text = data.text
        return text
        
class WebQATICandidateDataset(Dataset):
    """Text + Image candidates"""
    def __init__(
        self,
        data: List[WebQACandidate],
        lineidx_fpath: str,
        images_fpath: str,
    ):
        self.data=data
        with open(lineidx_fpath, "r") as fp_lineidx:
            self.lineidxs = [int(i.strip()) for i in fp_lineidx.readlines()]
        self.images_fpath = images_fpath
    
    def load_image(self, img_id: str) -> Image.Image:
        with open(self.images_fpath, "r") as fp:
            fp.seek(self.lineidxs[int(img_id)%10000000])
            imgid, img_base64 = fp.readline().strip().split('\t')
        im = Image.open(BytesIO(base64.b64decode(img_base64)))    
        return im

    def __getitem__(self, idx):
        data = self.data[idx]
        text = data.text
        image = self.load_image(data.id)
        return {
            "text": text,
            "image": image
        }
        

# class WebQADataset(Dataset):
#     def __init__(
#         self,
#         data_fpath: str,
#         lineidx_fpath: str,
#         images_fpath: str,
#         seed: int = 42
#     ):
#         self.datas = load_webqa_dataset(data_fpath)
#         with open(lineidx_fpath, "r") as fp_lineidx:
#             self.lineidxs = [int(i.strip()) for i in fp_lineidx.readlines()]

#         self.images_fpath = images_fpath

#     def load_image(self, image_data: WebQAImageData) -> Image.Image:
#         img_id = image_data.image_id
#         with open(self.images_fpath, "r") as fp:
#             fp.seek(self.lineidxs[int(img_id)%10000000])
#             imgid, img_base64 = fp.readline().strip().split('\t')
#         im = Image.open(BytesIO(base64.b64decode(img_base64)))    
#         return im
    
#     def __getitem__(self, idx):
#         sample = self.datas[idx]
#         image = self.load_image(sample.image)

#         candidate_text = "\n".join(
#             [sample.image.title, sample.image.caption]
#         )
#         return {
#             "query": sample.Q,
#             "candidate_text": candidate_text,
#             "candidate_image": image
#         }

#     def __len__(self):
#         return len(self.datas)