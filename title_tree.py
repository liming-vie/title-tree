# -*- coding: utf-8 -*-

__author__ = "Liming Huang (liming-vie)"
__email__ = "liming.vie@gmail.com"

import re
from collections import namedtuple
from functools import cmp_to_key


def make_title_patterns():
    chinese_values = '([一二三四五六七八九十百千零]+)'
    digit_values = '[1-9][0-9]*'
    values = ' ?(?P<val>(' + chinese_values + \
        '|([a-z])|([A-Z])|(' + digit_values + '))) ?'
    pts = [
        '(?P<prefix>第) ?' + values + ' ?(?P<suffix>(节|章|条))',
        '(?P<prefix>议案)(?P<val>'+chinese_values+')',
        '(?P<prefix>（?)' + values + '(?P<suffix>）)',
        '(?P<prefix>\(?)' + values + '(?P<suffix>\))',
        chinese_values + '(?P<suffix>[、\.])',
        '(?P<val>' + digit_values + '(((\.' + digit_values + ')+)|(?P<suffix>\.?)))',
    ]
    return map(lambda x: re.compile(x), pts)


TITLE_PATTERNS = make_title_patterns()


TitleInfo = namedtuple("TitleInfo", "val tpl origin length lnum idx")

Position = namedtuple("LinePostion", "lnum idx")


class Title:
    def __init__(self, title):
        self.info = title  # TitleInfo
        self.sub_node = None  # TreeNode
        self.end_pos = None  # Position
        self.lines = None  # list of text lines

    @property
    def id(self):  # 四、-> 4
        return self.info.val

    @property
    def title_text(self):
        return self.info.origin

    @property
    def start_pos(self):  # Position
        return Position(self.info.lnum, self.info.idx)


class DocTreeIterator:
    def __init__(self, root, lines, stack=[], pos=None):
        self._root = root
        self._lines = lines
        self._stack = stack
        if not pos:
            pos = (root, 0)
        self._move, self._set_idx = pos

    def get_all(self):
        return self._move.title_sets[self._set_idx]

    def get(self, i):
        title_sets = self._move.title_sets[self._set_idx]
        if i >= len(title_sets):
            return None
        return title_sets[i]

    def size(self):
        return len(self._move.title_sets[self._set_idx])

    def parent(self):
        if not self._stack:
            return None, None
        node, sidx, tidx = self._stack[-1]
        return self._make_iter(self._stack[:-1], (node, sidx)), tidx

    def child(self, i):
        title_sets = self._move.title_sets[self._set_idx]
        if i >= len(title_sets):
            return None

        sub_node = title_sets[i].sub_node
        if sub_node:
            stack = self._stack + [(self._move, self._set_idx, i)]
            pos = (sub_node, 0)
            return self._make_iter(stack, pos)
        return None

    def first(self):
        return self._make_iter(self._stack, (self._move, 0))

    def last(self):
        idx = self._move.size() - 1
        return self._make_iter(self._stack, (self._move, idx))

    def next(self):
        set_idx = self._set_idx+1
        if set_idx == self._move.size():
            return None
        return self._make_iter(self._stack, (self._move, set_idx))

    def prev(self):
        set_idx = self._set_idx-1
        if set_idx == self._move.size():
            return None
        return self._make_iter(self._stack, (self._move, set_idx))

    def _make_iter(self, stack, move):
        return DocTreeIterator(self._root, self._lines, stack, move)


def is_alpha(c):
    cval = ord(c)
    return (cval >= ord('a') and cval <= ord('z')) or (cval >= ord('A') and cval <= ord('Z'))


CHINESE_TO_VALUE_MAP = {'千': 1000, '百': 100, '十': 10, '九': 9, '八': 8, '七': 7,
                        '六': 6, '五': 5, '四': 4, '三': 3, '二': 2, '一': 1, '零': 0}


def string_to_value(valstr):
    if valstr[0].isdigit():
        if valstr.isdigit():
            return int(valstr), '1'
        return int(valstr.rstrip('.').split('.')[-1]), '1'

    if is_alpha(valstr[0]):
        cval = ord(valstr[0])
        return (cval-ord('a'), 'a') if valstr.islower() else (cval-ord('A'), 'A')

    i, val = 0, 0
    while i < len(valstr):
        v = CHINESE_TO_VALUE_MAP[valstr[i]]
        if i+1 < len(valstr) and CHINESE_TO_VALUE_MAP[valstr[i+1]] >= 10:
            v *= CHINESE_TO_VALUE_MAP[valstr[i+1]]
            i += 1
        i += 1
        val += v
    return val, '一'


class TreeNode:
    def __init__(self, title_sets=[]):
        self.title_sets = title_sets

    def start_pos(self, i):
        return self.title_sets[i][0].start_pos

    def end_pos(self, i):
        return self.title_sets[i][-1].end_pos

    def size(self):
        return len(self.title_sets)

    def print_tree(self, prefix=''):
        for i, title_set in enumerate(self.title_sets):
            print (prefix, '----- ' + str(i) + ' -----')
            for t in title_set:
                print(prefix, t.title_text, t.info.tpl, t.info.lnum, t.info.idx)
                if t.sub_node:
                    t.sub_node.print_tree(prefix+'\t')


class DocTree:
    def __init__(self, filepath, only_match_start=False):
        self.filepath = filepath
        self._match_start=only_match_start
        self.doc_lines = open(filepath).readlines()
        self._tpl_level = {}
        self.root = self._construct()

    def get_iter(self):
        return DocTreeIterator(self.root, self.doc_lines)

    def find(self, start, end):
        return self._find(start, end, [], self.root)

    def _find(self, start, end, stack, root):
        def _inside(start, end, cur_start, cur_end):
            return ((cur_start.lnum < start.lnum or
                     (cur_start.lnum == start.lnum and cur_start.idx <= start.idx)) and
                    (cur_end.lnum > end.lnum or
                     (cur_end.lnum == end.lnum and cur_end.idx >= end.idx)))
        for sidx, st in enumerate(root.title_sets):
            if _inside(start, end, root.start_pos(sidx), root.end_pos(sidx)):
                for tidx, t in enumerate(st):
                    if _inside(start, end, t.start_pos, t.end_pos):
                        if t.sub_node:
                            ret = self._find(
                                start, end, stack+[(root, sidx, tidx)], t.sub_node)
                            if ret[0]:
                                return ret
                        return DocTreeIterator(self.root, self.doc_lines, stack, (root, sidx)), tidx
        return None, None

    def _convert_info_to_lines(self, info):
        if not info:
            return None
        if info.lines.slnum == info.lines.elnum:
            lines = [self.doc_lines[info.lines.slnum]
                     [info.lines.sidx: info.lines.eidx]]
        else:
            lines = [self.doc_lines[info.lines.slnum][info.lines.sidx:]]
            lines += self.doc_lines[info.lines.slnum+1: info.lines.elnum]
            lines.append(self.doc_lines[info.lines.elnum][:info.lines.eidx])
        return info._replace(lines=lines)

    def _get_tpl_level(self, tpl):
        return self._tpl_level.get(tpl, 100)

    def _get_tree_height(self, root):
        depth = 0
        for st in root.title_sets:
            for t in st:
                if t.sub_node:
                    depth = max(depth, self._get_tree_height(t.sub_node))
        return depth + 1

    def _rebuild_tpl_level(self, root, depth, max_depth):
        if depth == 0:
            self._tpl_level.clear()

        for st in root.title_sets:
            for t in st:
                if t.sub_node:
                    self._rebuild_tpl_level(t.sub_node, depth + 1, max_depth)

        for st in root.title_sets:
            tpl = st[-1].info.tpl
            level = 100 - (max_depth - depth)
            if tpl not in self._tpl_level:
                self._tpl_level[tpl] = level
            else:
                self._tpl_level[tpl] = max(level, self._tpl_level[tpl])

    def _get_all_titles(self):
        ret = []
        for pt in TITLE_PATTERNS:
            for lnum, line in enumerate(self.doc_lines):
                idx = 0
                while idx < len(line) and line[idx].isspace():
                    idx += 1
                start_idx = idx
                while idx < len(line):
                    if self._match_start:
                        if idx != start_idx:
                            break
                        mt = pt.match(line, idx)
                    else:
                        mt = pt.search(line, idx)
                    if not mt:
                        break

                    idx = mt.start()
                    groups = mt.groupdict()
                    val, vtype = string_to_value(groups['val'])
                    tpl = vtype

                    if (('prefix' not in groups) or (not groups['prefix'])) and idx > 0:
                        if ((vtype in ['a', 'A'] and is_alpha(line[idx-1])) or (vtype == '1' and line[idx-1].isdigit())):
                            idx = mt.end()
                            continue

                    if 'suffix' in groups and groups['suffix']:
                            tpl += groups['suffix']
                    elif vtype == '1':
                        l = len(groups['val'].split('.'))
                        if l > 1:
                            tpl += '.' + str(l)
                        elif idx != start_idx:
                            idx = mt.end()
                            continue

                    if 'prefix' in groups:
                        tpl = groups['prefix'] + tpl

                    ret.append(TitleInfo(val, tpl,
                                         origin=mt.group(),
                                         length=len(mt.group()),
                                         lnum=lnum,
                                         idx=idx)
                               )
                    idx = mt.end()

        start_title = TitleInfo(0, '', '', 0, 0, 0)
        if not ret:
            return [start_title]

        ret = [start_title] + ret
        ret.sort(key=cmp_to_key(lambda a, b: (
            a.idx-b.idx if a.lnum == b.lnum else a.lnum-b.lnum)))

        # remove consecutive titles
        idx, out_idx = 0, 0
        while idx < len(ret):
            j = idx + 1
            while j < len(ret):
                if (ret[j].tpl == ret[j-1].tpl and
                    ret[j].lnum == ret[j-1].lnum and
                        ret[j].idx == ret[j-1].idx + ret[j-1].length):
                    j += 1
                else:
                    break
            if j == idx+1:
                ret[out_idx] = ret[idx]
                out_idx += 1
            idx = j
        ret = ret[:out_idx]

        return ret

    def _convert_title_to_title_result_info(self, titles, root, end=None):
        if not end:
            end = Position(len(self.doc_lines)-1, len(self.doc_lines[-1]))

        for si, st in enumerate(root.title_sets):
            for i, t in enumerate(st):
                if i < len(st)-1:
                    end_pos = st[i+1].info
                elif si < root.size() - 1:
                    end_pos = root.title_sets[si+1][0].info
                else:
                    end_pos = end

                if end_pos.idx == 0:
                    end_pos = Position(end_pos.lnum - 1,
                                       len(self.doc_lines[end_pos.lnum - 1]))

                st[i].end_pos = Position(end_pos.lnum, end_pos.idx-1)
                start_pos = Position(t.info.lnum, t.info.idx)
                if start_pos.lnum == end_pos.lnum:
                    lines = [self.doc_lines[start_pos.lnum]
                             [start_pos.idx: end_pos.idx]]
                else:
                    lines = [self.doc_lines[start_pos.lnum][start_pos.idx:]]
                    lines += self.doc_lines[start_pos.lnum + 1: end_pos.lnum]
                    lines.append(self.doc_lines[end_pos.lnum][:end_pos.idx])
                st[i].lines = lines

                if t.sub_node:
                    self._convert_title_to_title_result_info(
                        titles, t.sub_node, end_pos)

    def _construct(self):
        root = TreeNode([[]])
        stack = [root]

        titles = self._get_all_titles()
        for idx, title in enumerate(titles):
            cur_node = stack[-1]
            title_set = cur_node.title_sets[-1]
            phase = Title(title)
            prev_title = None if idx == 0 else titles[idx - 1]
            cur_level = self._get_tpl_level(title.tpl)
            if idx == 0:
                title_set.append(phase)
            elif title.tpl == prev_title.tpl:
                if title.val == prev_title.val+1:
                    title_set.append(phase)
                else:
                    cur_node.title_sets.append([phase])
            else:
                prev_level = self._get_tpl_level(prev_title.tpl)
                if title.val == 0 and prev_level == cur_level:
                    cur_node.title_sets.append([phase])
                elif prev_level < cur_level:
                    subnode = TreeNode([[phase]])
                    cur_node.title_sets[-1][-1].sub_node = subnode
                    stack.append(subnode)
                else:
                    while stack:
                        flag = False
                        i = len(stack) - 1
                        while i >= 0:
                            set_idx = stack[i].size() - 1
                            while set_idx >= 0:
                                cur_set = stack[i].title_sets[set_idx]
                                last_title = cur_set[-1].info
                                if last_title.tpl == title.tpl and last_title.val + 1 == title.val:
                                    subsets = []
                                    if cur_set[-1].sub_node:
                                        subsets = cur_set[-1].sub_node.title_sets
                                    else:
                                        cur_set[-1].sub_node = TreeNode()
                                    for st in range(set_idx+1, stack[i].size()):
                                        subsets.append(stack[i].title_sets[st])
                                    cur_set[-1].sub_node.title_sets = subsets
                                    stack[i].title_sets[set_idx].append(phase)
                                    stack[i].title_sets = stack[i].title_sets[:set_idx+1]
                                    flag = True
                                    break
                                set_idx -= 1
                            if flag:
                                stack = stack[:i+1]
                                break
                            i -= 1
                        if flag:
                            break

                        cur_node = stack[-1]
                        prev_node_title = None if len(
                            stack) < 2 else stack[-2].title_sets[-1][-1]
                        last_title = cur_node.title_sets[-1][-1]
                        last_level = self._get_tpl_level(last_title.info.tpl)
                        if len(stack) == 1 or last_level == cur_level:
                            if last_title.info.tpl == title.tpl and last_title.info.val + 1 <= title.val:
                                cur_node.title_sets[-1].append(phase)
                            else:
                                cur_node.title_sets.append([phase])
                            break
                        elif (len(stack) > 1 and
                              cur_level < last_level and
                              cur_level > self._get_tpl_level(prev_node_title.info.tpl)):
                            cur_node.title_sets.append([phase])
                            break
                        stack.pop()
                    self._rebuild_tpl_level(
                        root, 0, self._get_tree_height(root)-1)

        self._convert_title_to_title_result_info(titles, root)
        # root.print_tree()
        return root


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python3 title_tree.py filepath')

    filepath = sys.argv[1]
    tree = DocTree(filepath, only_match_start=True)

    def print_res(res, prefix):
        print (prefix, res.title_text, res.start_pos,
               res.end_pos, res.lines, '\n')

    # print tree structure
    tree.root.print_tree()

    # find the node that cover the range [start, end]
    iter, i = tree.find(Position(1, 0), Position(1, 0))
    while iter:
        print_res(iter.get(i), "")
        iter, i = iter.parent()

    def traversal(iter, prefix):
        while iter:
            res = iter.get_all()
            for idx, t in enumerate(res):
                print_res(t, prefix)
                traversal(iter.child(idx), prefix+'\t')
            iter = iter.next()

    # traversal the whole tree from first one
    # iter = tree.get_iter()
    # traversal(iter, "")
