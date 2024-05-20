# @description:
# @author:Jianping Zhou
# @email:jianpingzhou0927@gmail.com
# @Time:2022/10/29 16:52
import math
import json


def read_source_target(sourcefile, targetfile):
    with open(sourcefile, 'r', encoding='utf-8') as source:
        source_content = source.readlines()
        source.close()
    with open(targetfile, 'r', encoding='utf-8') as target:
        target_content = target.readlines()
        target.close()
    print(sourcefile, len(source_content), len(target_content))
    return source_content, target_content


def list_slice(alist, size, index):
    size = math.ceil(len(alist) / size)
    start = size * index
    end = (index + 1) * size if (index + 1) * size < len(alist) else len(alist)
    return alist[start:end]


def read_json(jsonfile):
    with open(jsonfile, "r", encoding='utf-8') as f:
        data = json.load(f)
        f.close()
    return data


def merge_train():
    # with open('tsdae_without_bginfo/train_0.source', 'r', encoding='utf-8') as f:
    #     print(len(f.readlines()))
    #     f.close()
    # with open('tsdae_without_bginfo/train_0.target', 'r', encoding='utf-8') as f:
    #     print(len(f.readlines()))
    #     f.close()
    data = read_json('tsdae_without_bginfo/traintarget_0.json')
    print(len(data))
    with open('tsdae_without_highlightcell/train_0.target', 'w+', encoding='utf-8') as f:
        for item in data:
            item = item.replace('\n', '').replace('\r', '')
            f.write(item + '\n')
        f.close()
    with open('tsdae_without_highlightcell/train_0.target', 'r', encoding='utf-8') as f:
        print(len(f.readlines()))
        f.close()


if __name__ == "__main__":
    with open('../../MyDataset/train/train.json', 'r', encoding='utf-8') as f:
        ALLdata = json.load(f)
    dataslice = list_slice(ALLdata, 10, 0)
    with open('tsdae_without_bginfo/train.source', 'w+', encoding='utf-8') as source:
        with open('tsdae_without_bginfo/train.target', 'w+', encoding='utf-8') as target:
            for i in range(10):
                # sourcefile = 'tempdata/source/train_{}.source'.format(i)
                # targetfile = 'tempdata/target/train_{}.target'.format(i)

                sourcefile = 'tsdae_without_bginfo/train_{}.source'.format(i)
                targetfile = 'tsdae_without_bginfo/train_{}.target'.format(i)

                # sourcefile = 'tsdae_without_bginfo_highlightcell/train_{}.source'.format(i)
                # targetfile = 'tsdae_without_bginfo_highlightcell/train_{}.target'.format(i)

                # sourcefile = 'tsdae_without_highlightcell/train_{}.source'.format(i)
                # targetfile = 'tsdae_without_highlightcell/train_{}.target'.format(i)
                sources, targets = read_source_target(sourcefile, targetfile)
                for i in range(len(sources)):
                    source.write(sources[i])
                    target.write(targets[i])
            source.close()
        target.close()
    with open('tsdae_without_bginfo/train.source', 'r', encoding='utf-8') as f:
        print(len(f.readlines()))
        f.close()
    with open('tsdae_without_bginfo/train.target', 'r', encoding='utf-8') as f:
        print(len(f.readlines()))
        f.close()
    # merge_train()
