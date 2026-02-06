
import os
import heapq
import warnings
import numpy as np
import ctypes,sys
from typing import Any, Iterator, List,Tuple
from collections.abc import Iterable
from numba import njit
def no_repeat_fast(arr):
    """"""
    if len(arr) == 0: return arr
    mask = np.concatenate(([True], arr[1:] != arr[:-1]))
    return arr[mask]
def iterable_pure(x):
    return isinstance(x, Iterable)
def iterable(x):  
    return not isinstance(x, str) and isinstance(x, Iterable) and len(x) > 0
def deep_copy(x):
    if isinstance(x, np.ndarray):
        res = []
        for i in range(len(x)):
            res.append(deep_copy(x[i]))
        return np.array(res, dtype=x.dtype)
    elif isinstance(x, list):
        res = []
        for i in range(len(x)):
            res.append(deep_copy(x[i]))
        return res
    else:
        return x
def get_deep(x):
    return get_deep(x[0]) + 1 if iterable(x) else 0
def deep_yield(x, keep_dim=0):
    if get_deep(x) > keep_dim:
        for i in x:
            yield from deep_yield(i, keep_dim)  
    else:
        yield x
def __deep_yield_simple(x):
    """
    :param x:
    :return:
    """
    if not isinstance(x, str) and isinstance(x, Iterable):
        for i in x:
            for y in deep_yield(i):
                yield y
    else:
        yield x
def deep_flatten2(x,fla_dict=True):
    res=[]
    for i in x:
        if isinstance(i,np.ndarray):
            i=i.tolist()
        if isinstance(i,list) or isinstance(i,tuple):
            res.extend(deep_flatten2(i))
        elif fla_dict and isinstance(i,dict):
            res.extend([deep_flatten2(j) for j in i.values()])
        else:
            res.append(i)
    return res
def deep_filter(x, func, keep_dim):
    """
    modifiable deep iter, keep traj satisfy func
    :param x:
    :param func: bool function
    :param keep_dim: dim the func need
    :return:
    """
    def _deep_filter(x, func, deep):
        if deep == keep_dim + 1:
            i = 0
            while i < len(x):
                if not func(x[i]):
                    del x[i]
                else:
                    i += 1
        else:
            for y in x:
                _deep_filter(y, func, deep - 1)
    deep = get_deep(x)
    _deep_filter(x, func, deep)
    return x
def deep_flatten(x, keep_dim=0):
    res = []
    for x in deep_yield(x, keep_dim):
        res.append(x)
    return res
def flatten(x: list):
    x = deep_copy(x)
    i = 0
    while i < len(x):
        if isinstance(x[i], list):
            v = x.pop(i)
            x.extend(v)
        else:
            i += 1
    return x
def flatten_times(x: list, t=1):
    x = deep_copy(x)
    for _ in range(t):
        x = flatten_fast(x)
    return x
def flatten_fast(x: list):
    """
    >>> flatten_fast([[1,2],[3]])
    [1,2,3]
    """
    return sum(x, [])
def list_reshape(li: list, shape: list):
    """
    reshape list
    list_reshape([1,2,3,4,5],[3]) => [[1,2,3],[4,5]]
    list_reshape([1,2,3,4,5,6,7,8],[2,2]) => [[[1,2],[3,4]],[[5,6],[7,8]]]
    :param li: list
    :param shape: shape
    :return:
    """
    def val(id):  
        return ctypes.cast(id, ctypes.py_object).value
    def stack(li, w):
        h = len(li) // w
        a, tail = li[:h * w], li[h * w:]
        adr = np.array([id(x) for x in a]).reshape([h, w])
        if w > 1:
            a = [[val(int(adr[j, i])) for i in range(w)] for j in range(h)]
        else:
            a = [val(int(adr[j, 0])) for j in range(h)]
        if len(tail) > 1: a.append(tail)
        return a
    if len(shape) < 1:
        return li
    for s in shape[1:]:
        assert isinstance(s, int), 'shape is not int'
    for w in shape[::-1]:
        li = stack(li, w)
    return li
def _test_lsit_reshape():
    a = ['as', 1, [1, 1], {1: 2, 3: 4}, 3.5, (1, 2), None, object]
    b = list_reshape(a, [4])
    assert b == [['as', 1, [1, 1], {1: 2, 3: 4}], [3.5, (1, 2), None, object]]
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    b = list_reshape(a, [2, 3])
    assert b == [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11]]]
    print(list_reshape([1, 2, 3, 4, 5, 6, 7, 8], [2, 2]))
    print(list_reshape([1, 2, 3, 4, 5], [3]))
def zipxs(l1, l2, *ls):
    if len(l1)==0 or len(l2)==0: return []
    if isinstance(l1[0], list):
        l = [[*i, j] for i in l1 for j in l2]
    else:
        l = [[i, j] for i in l1 for j in l2]
    if len(ls) == 0: return l
    return zipxs(l, ls[0], *ls[1:])
def T(x):return zip(*x)
def partition(x, parts):
    assert sum(parts) <= 1
    assert all([p >= 0 for p in parts])
    res = []
    last = 0
    for p in parts:
        l = int(len(x) * p)
        res.append(x[last:last + l])
        last += l
    return res
def reverse2(x: list):
    """
    [ [1,2,3],[1,2] ] to [ [1,1],[2,2],[3] ]
    :param x: list^2, H*W
    :return: list of W*H
    """
    H, W = len(x), max([len(i) for i in x])
    res = [[] for i in range(W)]  
    for h in range(H):
        for w in range(len(x[h])):
            res[w].append(x[h][w])
    return res
def shuffle(data, *others):
    data = [data, *others]
    for d in data:
        assert iterable(d)
        assert len(d) == len(data[0])
    idx = np.array([i for i in range(len(data[0]))])
    np.random.shuffle(idx)
    for dat in data:
        for i in range(len(idx)):
            dat[i], dat[idx[i]] = dat[idx[i]], dat[i]
    return data if len(others) > 0 else data[0]
def batch_iter(bs, data, *others):
    data = [data, *others]
    for d in data:
        assert iterable(d)
        assert len(d) == len(data[0])
    for i in range(len(data[0]) // bs):
        batch = [dat[i * bs:min(len(data[0]), (i + 1) * bs)] for dat in data]
        yield batch if len(others) > 0 else batch[0]
def topki(a, k): 
    assert k<=len(a)
    assert isinstance(a,list) or isinstance(a,np.ndarray)
    islist=isinstance(a,list)
    b=np.array(a) if islist else a
    assert len(b)>0
    xi=np.argwhere(b>=min(heapq.nlargest(k, b))).reshape(-1)
    if islist: return list(xi)
    return xi
def topk(a, k): 
    assert k<=len(a)
    if isinstance(a,list):
        return heapq.nlargest(k, a,a.__getitem__)
    if isinstance(a,np.ndarray) or "<class 'numpy.ndarray'>" in str(type(a)):
         return heapq.nlargest(k, a)
    raise 'unsupport dtype'
def bisearch_lambda_list(arr, fun):
    """return idx, fun([0,idx])=True"""
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        ok=fun(mid)
        if ok: left = mid + 1 
        else: right = mid - 1 
    return min(left,right)
def choice(data:List,num:int,replace=False, p=None):
    """
    datanum
    :param data:
    :param num: 0<=num<=len(traj)
    :param replace: True
    :param p: sum=1
    :return:
    """
    assert iterable_pure(data)
    if num>len(data) and replace==False:
        replace=True
        warnings.warn('in choice, num>len(data), we set replace=True')
    mask=np.random.choice(len(data),num,replace=replace,p=p)
    is_list=isinstance(data,list) or isinstance(data,tuple)
    dat=np.array(data,dtype=object) if is_list else data.copy()
    dat=dat[mask]
    return list(dat) if is_list else dat
