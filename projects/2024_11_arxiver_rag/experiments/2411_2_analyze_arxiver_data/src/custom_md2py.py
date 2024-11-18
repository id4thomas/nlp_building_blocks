import re
from typing import List, Optional

from markdown import markdownFromFile, markdown
from bs4 import BeautifulSoup
from bs4.element import Tag

class TreeOfContents:
    """Tree abstraction for markdown source"""

    source_type = BeautifulSoup
    ## Attributes
    valid_tags = ('a', 'abbr', 'address', 'area', 'article', 'aside', 'audio',
        'b', 'base', 'bdi', 'bdo', 'blockquote', 'body', 'br', 'button',
        'canvas', 'caption', 'cite', 'code', 'col', 'colgroup', 'data',
        'datalist', 'dd', 'del', 'details', 'dfn', 'dialog', 'div', 'dl', 'dt',
        'em', 'embed', 'fieldset', 'figcaption', 'figure', 'footer', 'form',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'head', 'header', 'hgroup', 'hr',
        'html', 'i', 'iframe', 'img', 'input', 'ins', 'kbd', 'keygen', 'label',
        'legend', 'li', 'link', 'main', 'map', 'mark', 'menu', 'menuitem',
        'meta', 'meter', 'nav', 'noscript', 'object', 'ol', 'optgroup',
        'option', 'output', 'p', 'param', 'picture', 'pre', 'progress', 'q',
        'rp', 'rt', 'ruby', 's', 'samp', 'script', 'section', 'select', 'small',
        'source', 'span', 'strong', 'style', 'sub', 'summary', 'sup', 'table',
        'tbody', 'td', 'template', 'textarea', 'tfoot', 'th', 'thead', 'time',
        'title', 'tr', 'track', 'u', 'ul', 'var', 'video', 'wbr')
    allowed_attrs = ('string', 'name')
    header_name_pattern = r"^h([1-6])"
    ## md->html
    default_md_extensions = ['markdown.extensions.fenced_code','markdown.extensions.tables']
     
    def __init__(self, source: Tag, branches=(), descendant_tags: List[Tag]=(), depth: Optional[int]=None):
        """
        Construct TreeOfContents object using source

        :param SourceType source: parsed source
        :param list TreeOfContents branches: list of direct children
        :param list SourceType descendants: all descendants
        """
        if source is None:
            raise ValueError('NoneType source passed into TreeOfContents')
        self.source = source
        self.depth = depth or self.parseTopDepth()
        # MODIFIED - make branches with source.children & expand descendants later
        # self.descendants = descendants or list(source.descendants)
        self.branches: List["TreeOfContents"] = branches or self.parseBranches(descendant_tags)
        self.descendants: List["TreeOfContents"] = self.expandDescendants()

    @classmethod
    def getHeadingLevel(cls, bs) -> Optional[int]:
        """
        >>> bsify = lambda html: BeautifulSoup(html, 'html.parser')
        >>> bs = bsify('<h1>Hello</h1>').h1
        >>> TOC.getHeadingLevel(bs)
        1
        >>> bs2 = bsify('<p>Hello</p>').p
        >>> TOC.getHeadingLevel(bs2)

        >>> bs3 = bsify('<article>Hello</article>').article
        >>> TOC.getHeadingLevel(bs3)

        """
        # MODIFIED - get header_num by pattern
        header_match = re.search(cls.header_name_pattern, bs.name)
        if header_match:
            try:
                header_num = int(header_match.group(1))
            except (ValueError, IndexError, TypeError):
                return None
            return header_num
        else:
            return None
        # try:
        #     return int(bs.name[1])
        # except (ValueError, IndexError, TypeError):
        #     return None

    def parseTopDepth(self) -> int:
        """
        Parse highest heading in markdown

        >>> TOC.fromHTML('<h2>haha</h2><h1>hoho</h1>').parseTopDepth()
        1
        >>> TOC.fromHTML('<h3>haha</h3><h2>hoho</h2>').parseTopDepth()
        2
        """
        for i in range(1, 7):
            if getattr(self.source, 'h{}'.format(i)):
                return i

    def expandDescendants(self) -> List[Tag]:
        """
        Expand descendants from list of branches

        :param list branches: list of immediate children as TreeOfContents objs
        :return: list of all descendants
        """
        descendants = []
        for b in self.branches:
            descendants.append(b)
            descendants.extend(b.expandDescendants())
        return descendants
        # return sum([b.descendants for b in self.branches], []) + \
            # [b.source for b in self.branches]

    def parseBranches(self, descendant_tags: List[Tag]) -> List["TreeOfContents"]:
        """
        Parse top level of markdown

        :param list elements: list of source objects
        :return: list of filtered TreeOfContents objects
        """
        # parsed, parent, cond = [], False, lambda b: (b.string or '').strip()
        # parsed, parent, cond = [], False, lambda b: b.name and b.name in self.valid_tags
        parent_level = self.getHeadingLevel(self.source)
        if parent_level is None:
            parent_level = 0

        parsed_branches = []
        cur_level = 7
        # loop through descendant tags
        cond = lambda b: b.name and b.name in self.valid_tags
        for branch in filter(cond, descendant_tags):
            level = self.getHeadingLevel(branch)
            if level and level<=cur_level:
                cur_level = level
                node = {'level': level, 'source': branch, 'descendants': []}
                parsed_branches.append(node)
            else:
                if not parsed_branches:
                    node = {'level': 7, 'source': branch, 'descendants': []}
                    parsed_branches.append(node)
                else:
                    parsed_branches[-1]['descendants'].append(branch)
        # print("PARSED_BRANCH", parsed_branches)
        ## Make TOC
        return [TOC(depth=self.depth+1, source=x['source'], descendant_tags=x['descendants']) for x in parsed_branches]

    def __getattr__(self, attr, *default):
        """Check source for attributes"""
        tag = attr[:-1]
        if attr=="source":
            return self.source
        if attr in self.allowed_attrs:
            return getattr(self.source, attr, *default)
        if attr in self.valid_tags:
            return next(filter(lambda t: t.name == attr, self.branches), None)
        if len(default):
            return default[0]
        if attr[-1] == 's' and tag in self.valid_tags:
            condition = lambda t: t.name == tag
            return filter(condition, self.branches)
        raise AttributeError("'TreeOfContents' object has no attribute '%s'" % attr)

    def __repr__(self):
        """Display contents"""
        # return str(self)
        # MODIIFED - return str(source) to access html
        return str(self.source)

    def __str__(self):
        """Display contents"""
        return self.string or ''

    def __iter__(self):
        """Iterator over children"""
        return iter(self.branches)

    def __getitem__(self, i):
        return self.branches[i]

    @classmethod
    def fromMarkdown(cls, md: str, *args, **kwargs):
        """
        Creates abstraction using path to file

        :param str path: path to markdown file
        :return: TreeOfContents object
        """
        if not kwargs.get('extensions', None):
            kwargs['extensions'] = cls.default_md_extensions
        md_text = markdown(md, *args, **kwargs)
        return TOC.fromHTML(md_text)

    @staticmethod
    def fromHTML(html: str, *args, **kwargs):
        """
        Creates abstraction using HTML

        :param str html: HTML
        :return: TreeOfContents object
        """
        source = BeautifulSoup(html, 'html.parser', *args, **kwargs)
        # parsed = []
        # parsed, parent, cond = [], False, lambda b: b.name and b.name in TreeOfContents.valid_tags
        # for branch in filter(cond, source.children):
        #     parsed.append({'root':branch.string, 'source':branch})
            
        # branches = [TOC(depth=2, **kwargs) for kwargs in parsed]
        return TOC(
            # '[document]',
            source=source,
            depth=0,
            # branches = branches
            descendant_tags=source.children
        )

TOC = TreeOfContents
def md2py(md, *args, **kwargs):
    """
    Converts markdown file Python object

    :param str md: markdown string
    :return: object
    """
    return TreeOfContents.fromMarkdown(md, *args, **kwargs)