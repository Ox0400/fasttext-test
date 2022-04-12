#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: zhipeng
# @Email: zhipeng.py@gmail.com
# @Date:   2022-04-12 19:38:42
# @Last Modified By:    zhipeng
# @Last Modified: 2022-04-12 23:07:37


# model = fasttext.train_supervised(input="base.train", epoch=10, lr=0.6, wordNgrams=2, loss='ova', minCount=1)


import fasttext, jieba

jieba.add_word('不好看')
jieba.add_word('难看')

train_labels = """
__label__颜色-不好看  这个 颜色 不好看 不是 难 这 也 太 难看 了 很 相当 难 看 非常 丑 丑得 要 死 了 真的 不好 看
__label__颜色-好看  这个颜色好看 好看 好好看 十分好看 好看的很 颜色好 好
"""

with open('base.train', 'wb') as f:
    f.write(train_labels)

model = fasttext.train_supervised(input="base.train", epoch=10, wordNgrams=2, minCount=0, verbose=True)



def get_lab(text):
    # l = " ".join(model.predict(u"这个颜色好好看", k=1)[0]).replace('__label__', '')
    s = " ".join(list(jieba.cut(text)))
    l = " ".join(model.predict(s, k=1)[0]).replace('__label__', '')
    print ("%s --> %s" % (s, l))
    return l

train_labels = """
__label__颜色-不好看  这个 颜色 不好看 不是 难 这 也 太 难看 了 很 相当 难 看 非常 丑 丑得 要 死 了 真的 不好 看
__label__颜色-好看  这个颜色好看 好看 好好看 十分好看 好看的很 颜色好 好
"""

assert get_lab(u"这个颜色好好看")  == u"颜色-好看"
assert get_lab(u"这个 颜色  好 好看")  == u"颜色-好看"
assert get_lab(u"这个 颜色 好好看")  == u"颜色-好看"
assert get_lab(u"这个 好好看 颜色")  == u"颜色-好看"
assert get_lab(u"这个 颜色 很 好看")  == u"颜色-好看"
assert get_lab(u"这个 颜色 不好看")  == u"颜色-好看" ### ???
assert get_lab(u"这个 颜色 不好 看")  == u"颜色-不好看"
assert get_lab(u"这个好看")  == u"颜色-好看"
assert get_lab(u"这个 好看")  == u"颜色-好看"
assert get_lab(u"我 感觉 这个 还 挺 好看")  == u"颜色-好看"
assert get_lab(u"难 看")  == u"颜色-不好看"
assert get_lab(u"难 看 死了")  == u"颜色-不好看"
assert get_lab(u"这个 不好 看")  == u"颜色-不好看"
assert get_lab(u"这个 好看 吗")  == u"颜色-好看"

assert get_lab(u"这个不好看吗")  == u"颜色-好看" ### ???
assert get_lab(u"这个不是很好看吗")  == u"颜色-好看" ### ???

assert get_lab(u"太假了") == u"颜色-好看" ## 默认??
