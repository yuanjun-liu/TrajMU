exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import json, time, os, sys
from typing import Any
from _tool.mThread import  ThreadLock, ProcessLock
import numpy as np
from _tool.mTime import now_date_time
from _tool.mFile import check_dir,log_dir
class LogJsonIdxs:
    def __init__(self, name, _flock=None,refresh=True,mode='r',backup=True) -> None:
        self.data: dict = dict()  
        self._file = name if '/' in name or '\\' in name else os.path.join(log_dir(), name) + '.json'
        self._plock = ProcessLock()
        self._tlock = ThreadLock()
        self._flock = _flock
        self._refresh=refresh
        self._mode=mode
        self._backup=backup
        self._backfile=self._file+'.bk'
        check_dir(self._file)
        if 'r' in self._mode:self.load()
    def save(self):
        assert 'w' in self._mode
        with open(self._file, 'w') as f:
            json.dump(self.data, f)
        if self._backup:
            with open(self._backfile, 'w') as f:
                json.dump(self.data, f)
    def load(self):
        assert 'r' in self._mode
        if os.path.exists(self._file):
            try:
                with open(self._file, 'r') as f:
                    self.data = json.load(f)
            except Exception as e:
                pass
    def __getitem__(self, idx):
        x = self.__getitem(idx)
        if x is not None: return x
        if self._refresh: 
            if self._flock: self._flock.acquire()
            with self._plock:
                with self._tlock:
                    self.load()
            if self._flock: self._flock.release()
        return self.__getitem(idx)
    def __getitem(self, idx): 
        if isinstance(idx, tuple):
            d: dict = self.data
            for i in idx:
                assert isinstance(i, str)
                if d is None: return None
                d = d.get(i, None)
            return d
        else:
            assert isinstance(idx, str)
            return self.data[idx]
    def __setitem__(self, idx, value):
        if self._flock: self._flock.acquire()
        with self._plock, self._tlock:
            if self._refresh: self.load()
            d = self.data
            if isinstance(idx, tuple):
                for i in idx[:-1]:
                    assert isinstance(i, str)
                    if i not in d:
                        d[i] = dict()  
                    if i in d and not isinstance(d[i], dict):
                        raise IndexError('warning: logger traj is covered')
                    d = d[i]
                assert isinstance(idx[-1], str)
                d[idx[-1]] = value
            else:
                assert isinstance(idx, str)
                d[idx] = value
            if self._refresh: self.save()
        if self._flock: self._flock.release()
class LogToTxt:
    def __init__(self, file, span=', ',new_file=True, _flock=None,csv=False,prefix=True,pt=True) -> None:
        """
        :param file: "a:/1.txt" or "1"
        :param span:
        :param new_file: True: delete existing file, False: append
        :param _flock:
        :param csv: file endswith .csv, span set to ,
        :param prefix: log time at head of each line
        """
        self.__file = file if '/' in file or '\\' in file else os.path.join(log_dir(),file)+('.csv' if csv else '.log')
        self.__span = span
        self.__prefix=prefix
        self._plock = ProcessLock()
        self._tlock = ThreadLock()
        self._flock = _flock
        self._pt=pt
        if self._flock: self._flock.acquire()
        check_dir(self.__file)
        if new_file:
            with self._plock:
                with self._tlock:
                    if os.path.exists(self.__file):
                        os.remove(self.__file)
        if self._flock: self._flock.release()
    def __call__(self, *args) -> Any:
        heads = [] 
        if self.__prefix is not None:
            if self.__prefix: heads.append(now_date_time())
            if isinstance(self.__prefix,str):heads.append(self.__prefix)
        head=self.__span.join(heads)
        if len(heads): head+='. '
        if self._flock: self._flock.acquire()
        with self._plock:
            with self._tlock:
                with open(self.__file, 'a', encoding='utf-8') as f:
                    f.write(head + self.__span.join([str(i) for i in args]) + '\n')
        if self._pt: print(head + self.__span.join([str(i) for i in args]) )
        if self._flock: self._flock.release()
    def print(self, *arg):
        self(arg)
    def error(self, *arg):
        self(arg, head='ERROR')
    def warning(self, *arg):
        self(arg, head='Warning')
class mPrintCapturer:  
    def __init__(self,log_name=None,out_err='out',pt=True):
        assert out_err in ['out','err'], 'un support type'
        self._out_err=out_err
        self._log=None if log_name is None else LogToTxt(log_name,pt=None,span='')
        self._pt=pt
        self._x=[]
    def _log_write(self):
        if len(self._x): self._log(*self._x)
        self._x.clear()
    def write(self, s:str): 
        if self._pt:
            if self._out_err=='out':
                sys.__stdout__.write(s)
                sys.__stdout__.flush()
            if self._out_err=='err':
                sys.__stderr__.write(s)
                sys.__stderr__.flush()
        if self._log: 
            while len(s) and '\n' in s:
                i=s.index('\n')
                x,s=s[:i],s[i+1:]
                self._x.append(x)
                self._log_write()
            if len(s):self._x.append(s)
    def __enter__(self): self.replace()
    def __exit__(self, exc_type, exc_val, exc_tb): self.restore()
    def replace(self):
        if self._out_err=='out':
            sys.stdout = self
        elif self._out_err=='err':
            sys.stderr = self
    def restore(self):
        self._log_write()
        sys.stdout = sys.__stdout__
        sys.stderr =sys.__stderr__
    def __del__(self): self.restore()
    def flush(self): self._log_write()
