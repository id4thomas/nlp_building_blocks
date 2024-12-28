from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env", env_file_encoding="utf-8", extra="ignore"
    )
    pipeline_src_dir: str
settings = Settings()

import sys
sys.path.append(settings.pipeline_src_dir)

from typing import List, Literal

import time
import traceback

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from core.data.paper import ArxivPaperSection, ArxivPaperMetadata, ArxivPaper
from core.parser.md2py import TreeOfContents

from modules.extractor.section_splitter import MarkdownArxivPaperSectionSplitter


from multiprocessing import Pool, Process
from functools import partial
NUM_PROCESSES = 8

def get_sections(idx, text):
    # row = df.iloc[idx]
    # text = row['markdown']
    # found_filter = False
    
    sections = None
    for filter_cls in MarkdownArxivPaperSectionSplitter.__subclasses__():
        try:
            if filter_cls.is_type(text):
                # print("FOUND",filter_cls)
                found_filter = True
                sections = filter_cls().split(text)
                break
        except RecursionError as e:
            print("{} RECURSION ERROR {}".format(idx, str(e)))
            return idx, None
        except Exception as e:
            print("{} ERROR {}".format(idx, str(e)))
            # print(traceback.format_exc())
            raise e
    return idx, sections

def main():
    texts = df.markdown.values.tolist()
    with Pool(NUM_PROCESSES) as pool:
        results = pool.starmap(get_sections, zip(range(df.shape[0]), texts))
    
    # Failed Count
    failed_count = sum(1 for _, sections in results if sections is None)
    print("Total {} failed {}".format(len(results), failed_count))

if __name__=="__main__":
    ## Load Sample
    df = pd.read_parquet("sample.parquet")
    print(df.shape, df.columns)
    start = time.time()
    main()
    end = time.time()
    print("Elapsed {:.3f}".format(end-start))