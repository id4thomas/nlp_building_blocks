from abc import ABC, abstractmethod
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter
from markdownify import markdownify as md

from core.data.paper import ArxivPaperSection
from core.parser.md2py import TreeOfContents

headers_to_split_on = [
    ("###", "h3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

class MarkdownArxivPaperSectionSplitter:
    def __init__(self):
        pass
    
    @abstractmethod
    def split(self, text: str) -> List[ArxivPaperSection]:
        raise NotImplementedError("split must be implemented")
    
    @classmethod
    def is_type(cls, text: str) -> bool:
        return False
    
    def get_children_sections(self, section: ArxivPaperSection) -> List[ArxivPaperSection]:
        text = section.text
        splits = markdown_splitter.split_text(text)
        children = []
        for split in splits:
            if 'h3' in split.metadata:
                header = "h3"
                title=split.metadata.get("h3", "")
            else:
                header = "p"
                title=""
                
            section = ArxivPaperSection(
                header=header,
                title=title,
                text=split.page_content
            )
            children.append(section)
        return children
    
class Case1Splitter(MarkdownArxivPaperSectionSplitter):
    """Case1: everything under h1 title"""
    
    @classmethod
    def is_type(cls, text: str) -> bool:
        toc = TreeOfContents.fromMarkdown(text)
        if len(toc.branches)==1 and toc.branches[0].name=="h1":
            return True
        return False
    
    def split(self, text: str) -> List[ArxivPaperSection]:
        toc = TreeOfContents.fromMarkdown(text)
        
        h1_child = list(toc.branches)[0]
        sections = []
        for child in h1_child.branches:
            # p - title, h6 - abstract
            if not child.name=="h2":
                continue
            section = ArxivPaperSection(
                header = child.name,
                title = child.getText(),
                text = child.getDescendantsMarkdown()
            )
            section.children = self.get_children_sections(section)
            sections.append(section)
        return sections
    
class Case2Splitter(MarkdownArxivPaperSectionSplitter):
    """Case2: title & contents in same level"""
    
    @classmethod
    def is_type(cls, text: str) -> bool:
        toc = TreeOfContents.fromMarkdown(text)
        if len(toc.branches)>1:
            return True
        return False
    
    def split(self, text: str) -> List[ArxivPaperSection]:
        toc = TreeOfContents.fromMarkdown(text)
        sections = []
        for child in toc.branches:
            # p - title, h6 - abstract
            if not child.name=="h2":
                continue
            
            section = ArxivPaperSection(
                header = child.name,
                title = child.getText(),
                text = child.getDescendantsMarkdown()
            )
            section.children = self.get_children_sections(section)
            sections.append(section)
        return sections