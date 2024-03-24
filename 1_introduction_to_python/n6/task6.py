def check(x: str, file: str):
    x = x.lower()
    d = dict()
    words = x.split()
    for i in range(len(words)):
        if d.get(words[i], -1) == -1:
            d[words[i]] = 1
        else:
            d[words[i]] += 1
    
    f = open(file, 'w')
    d = dict(sorted(d.items()))
    for word, count in d.items():
        print(word, count, file = f)
    f.close()


